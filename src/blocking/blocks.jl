
abstract type TaskBlock{V <: AbstractArray} end

array_type(::TaskBlock{V}) where {V} = V
Base.eltype(::TaskBlock{V}) where {V} = eltype(V)


"""
    LocalTaskBlock{V, Size}

Container of `Size` and variables of type `V`, part of a [`BlockGrid`](@ref). One block can run all
solver steps independantly of all other blocks, apart from those which require updating ghost cells.
"""
mutable struct LocalTaskBlock{V, Size <: BlockSize} <: TaskBlock{V}
    @atomic state :: BlockState  # TODO: atomic??
    size   :: Size               # Size (in cells) of the block
    pos    :: CartesianIndex{2}  # Position in the local block grid
    neighbours :: Neighbours{TaskBlock}
    mirror :: LocalTaskBlock     # Host/Device mirror of this block. If device is host, then it is a self reference
    # TODO: device? storing the associated GPU stream, or CPU cores (or maybe not, to allow relocations?)
    # Cells variables
    x      :: V
    y      :: V
    ρ      :: V
    u      :: V
    v      :: V
    E      :: V
    p      :: V
    c      :: V
    g      :: V
    uˢ     :: V
    pˢ     :: V
    work_1 :: V
    work_2 :: V
    work_3 :: V
    work_4 :: V
    mask   :: V  # TODO: remove ??

    function LocalTaskBlock{V, Size}(size::Size, pos; kwargs...) where {V, Size}
        # `neighbours` and `mirror` are set afterwards, when all blocks are created.
        block = new{V, Size}(Done, size, pos)

        b_size = block_size(size)
        for var in block_vars()
            label = string(var)
            var_array = V(undef, prod(b_size); alloc_array_kwargs(; label, kwargs...)...)
            setfield!(block, var, var_array)
        end

        return block
    end
end


block_size(blk::LocalTaskBlock) = block_size(blk.size)
real_block_size(blk::LocalTaskBlock) = real_block_size(blk.size)
ghosts(blk::LocalTaskBlock) = ghosts(blk.size)
state(blk::LocalTaskBlock) = blk.state

block_vars() = (:x, :y, :ρ, :u, :v, :E, :p, :c, :g, :uˢ, :pˢ, :work_1, :work_2, :work_3, :work_4, :mask)
main_vars()  = (:x, :y, :ρ, :u, :v, :E, :p, :c, :g, :uˢ, :pˢ)  # Variables synchronized between host and device
saved_vars() = (:x, :y, :ρ, :u, :v,     :p                  )  # Variables saved to/read from I/O
comm_vars()  = (        :ρ, :u, :v, :E, :p, :c, :g          )  # Variables exchanged between ghost cells

block_vars(blk::LocalTaskBlock) = getfield.(Ref(blk), block_vars())
main_vars(blk::LocalTaskBlock)  = getfield.(Ref(blk), main_vars())
saved_vars(blk::LocalTaskBlock) = getfield.(Ref(blk), saved_vars())
comm_vars(blk::LocalTaskBlock)  = getfield.(Ref(blk), comm_vars())


function Base.copyto!(dst_blk::LocalTaskBlock{V, Size}, src_blk::LocalTaskBlock{V, Size}) where {V, Size}
    if state(dst_blk) != Done || state(src_blk) != Done
        error("Both destination and source blocks must be `Done` to copy")
    elseif block_size(dst_blk) != block_size(src_blk)
        error("Destination and source blocks have different sizes: $(block_size(dst_blk)) != $(block_size(src_blk))")
    end

    # TODO: ensure this is done in the correct thread ??
    for (dst_var, src_var) in zip(main_vars(dst_blk), main_vars(src_blk))
        copyto!(dst_var, src_var)
    end
end


function Base.copyto!(dst_blk::LocalTaskBlock{A, Size}, src_blk::LocalTaskBlock{B, Size}) where {A, B, Size}
    if state(dst_blk) != Done || state(src_blk) != Done
        error("Both destination and source blocks must be `Done` to copy")
    elseif block_size(dst_blk) != block_size(src_blk)
        error("Destination and source blocks have different sizes: $(block_size(dst_blk)) != $(block_size(src_blk))")
    end

    # TODO: ensure this is done in the correct thread ?? + device??
    for (dst_var, src_var) in zip(main_vars(dst_blk), main_vars(src_blk))
        KernelAbstractions.copyto!(dst_blk.device, dst_var, src_var)
    end
end


function device_to_host!(blk::LocalTaskBlock)
    blk === blk.mirror && return  # device is host

    first_var = getfield(blk, first(block_vars()))
    if KernelAbstractions.get_backend(first_var) isa KernelAbstractions.GPU
        copyto!(blk.mirror, blk)
    else
        copyto!(blk, blk.mirror)
    end
end


function host_to_device!(blk::LocalTaskBlock)
    blk === blk.mirror && return  # device is host

    first_var = getfield(blk, first(block_vars()))
    if KernelAbstractions.get_backend(first_var) isa KernelAbstractions.GPU
        copyto!(blk, blk.mirror)
    else
        copyto!(blk.mirror, blk)
    end
end


function Base.show(io::IO, blk::LocalTaskBlock)
    pos_str = join(Tuple(blk.pos), ',')
    size_str = join(block_size(blk), '×')
    cell_count = prod(block_size(blk))
    print(io, "LocalTaskBlock(at ($pos_str) of size $size_str (total: $cell_count), state: $(state(blk)))")
end

#
# RemoteTaskBlock
#

"""
    RemoteTaskBlock{B}

Block located at the border of a [`BlockGrid`](@ref), containing MPI buffer of type `B` for
communication with other [`BlockGrid`](@ref)s.
"""
mutable struct RemoteTaskBlock{B} <: TaskBlock{B}
    @atomic state :: BlockState      # `Waiting` if comms are in progress, `Done` otherwise. TODO: atomic??
    pos        :: CartesianIndex{2}  # Position in the local block grid
    neighbour  :: LocalTaskBlock     # Remote blocks are on the edges of the sub-domain: there can only be one real neighbour
    rank       :: Int
    global_pos :: CartesianIndex{2}  # Rank position in the Cartesian process grid
    send_buf   :: MPI.Buffer{B}
    recv_buf   :: MPI.Buffer{B}
    send_req   :: MPI.AbstractRequest
    recv_req   :: MPI.AbstractRequest

    function RemoteTaskBlock{B}(size, pos, rank, global_pos, comm) where {B}
        # `neighbour` is set afterwards, when all blocks are created.
        block = new{B}(Done, pos)
        block.rank = rank
        block.global_pos = global_pos
        block.send_buf = MPI.Buffer(B(undef, size))
        block.recv_buf = MPI.Buffer(B(undef, size))
        block.send_req = MPI.Send_init(block.send_buf, comm; dest=rank)
        block.recv_req = MPI.Recv_init(block.recv_buf, comm; source=rank)
        return block
    end

    function RemoteTaskBlock{B}(pos) where {B}
        # Constructor for an non-existant task block, found at the edges of the global domain, where
        # there is no MPI rank.
        block = new{B}(Done, pos)
        block.rank = -1
        block.global_pos = CartesianIndex(0, 0)
        block.send_buf = MPI.Buffer(B(undef, 0))
        block.recv_buf = MPI.Buffer(B(undef, 0))
        block.send_req = MPI.Request()
        block.recv_req = MPI.Request()
        return block
    end
end

state(block::RemoteTaskBlock) = block.state

function Base.show(io::IO, blk::RemoteTaskBlock)
    pos_str = join(Tuple(blk.pos), ',')
    if blk.rank == -1
        print(io, "RemoteTaskBlock(at ($pos_str), to: nowhere, state: $(state(blk)))")
    else
        global_str = join(Tuple(blk.global_pos), ',')
        print(io, "RemoteTaskBlock(at ($pos_str), to: process $(blk.rank) at ($global_str), state: $(state(blk)))")
    end
end
