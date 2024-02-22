
@generic_kernel function boundary_conditions!(
    ρ::V, u::V, v::V, p::V, c::V, g::V, E::V,
    bsize::BlockSize, axis::Axis, side::Side,
    u_factor::T, v_factor::T
) where {T, V <: AbstractArray{T}}
    @kernel_init begin
        # `incr` is the stride of `axis` going towards the edge along `side`
        incr = stride_along(bsize, axis)
        incr = ifelse(side in first_sides(), -incr, incr)
    end

    i = @index_2D_lin()
    ig = i + incr  # index of the ghost cell

    # TODO: use `comm_vars()` and iterate it like `pack_to_array` ?
    for _ in 1:ghosts(bsize)
        ρ[ig] = ρ[i]
        u[ig] = u[i] * u_factor
        v[ig] = v[i] * v_factor
        p[ig] = p[i]
        c[ig] = c[i]
        g[ig] = g[i]
        E[ig] = E[i]

        i  -= incr
        ig += incr
    end
end


function boundary_conditions!(params::ArmonParameters{T}, blk::LocalTaskBlock, side::Side) where {T}
    (u_factor::T, v_factor::T) = boundary_condition(params.test, side)
    domain = border_domain(blk.size, side)
    boundary_conditions!(params, blk, domain, blk.size, params.current_axis, side, u_factor, v_factor)
end


@kernel_function function vars_ghost_exchange(
    vars₁::NTuple{N, V}, i₁, ig₁, sg₁,
    vars₂::NTuple{N, V}, i₂, ig₂, sg₂,
    ghosts
) where {N, V}
    for gᵢ in 0:ghosts-1
        j₁ = gᵢ * sg₁
        j₂ = gᵢ * sg₂

        # Real cells of 1 to ghosts of 2
        # TODO: KernelAbstractions.Extras.@unroll ??
        for v in 1:N
            vars₂[v][ig₂ + j₂] = vars₁[v][i₁ + j₁]
        end

        # Real cells of 2 to ghosts of 1
        # TODO: KernelAbstractions.Extras.@unroll ??
        for v in 1:N
            vars₁[v][ig₁ + j₁] = vars₂[v][i₂ + j₂]
        end
    end
end


@generic_kernel function block_ghost_exchange(
    vars₁::NTuple{N, V},
    vars₂::NTuple{N, V},
    bsize::BlockSize, axis::Axis, side₁::Side
) where {N, V}
    @kernel_init begin
        side₂ = opposite_of(side₁)

        # Offsets to first ghost cell
        sg  = stride_along(bsize, axis)
        sg₁ = ifelse(side₁ in first_sides(), -sg, sg)
        sg₂ = ifelse(side₂ in first_sides(), -sg, sg)

        # Convertion from `i₁` to `i₂`, exploiting the fact that both blocks have the same size
        d₂ = stride_along(bsize, axis) * (real_size_along(bsize, axis) - 1)
        d₂ = ifelse(side₂ in first_sides(), -d₂, d₂)
    end

    i₁ = @index_2D_lin()
    i₂ = i₁ + d₂

    ig₁ = i₁ + sg₁
    ig₂ = i₂ + sg₂

    vars_ghost_exchange(
        vars₁, i₁, ig₁, sg₁,
        vars₂, i₂, ig₂, sg₂,
        ghosts(bsize)
    )
end


function block_ghost_exchange(
    params::ArmonParameters, blk₁::LocalTaskBlock{V, Size}, blk₂::LocalTaskBlock{V, Size}, side::Side
) where {V, Size <: StaticBSize}
    # Exchange between two blocks with the same dimensions
    domain = border_domain(blk₁.size, side)
    # println()
    # @show blk.pos params.current_axis side domain blk.size stride_along(blk.size, params.current_axis) (side in first_sides())
    block_ghost_exchange(params, domain,
        comm_vars(blk₁), comm_vars(blk₂),
        blk₁.size, params.current_axis, side
    )
end


@generic_kernel function block_ghost_exchange(
    vars₁::NTuple{N, V}, bsize₁::BlockSize,
    vars₂::NTuple{N, V}, bsize₂::BlockSize,
    axis::Axis, side₁::Side
) where {N, V}
    @kernel_init begin
        side₂ = opposite_of(side₁)

        # Offsets to first ghost cell
        sg₁ = stride_along(bsize₁, axis)
        sg₂ = stride_along(bsize₂, axis)
        sg₁ = ifelse(side₁ in first_sides(), -sg₁, sg₁)
        sg₂ = ifelse(side₂ in first_sides(), -sg₂, sg₂)
    end

    i₁ = @index_2D_lin()

    # `bsize₁` and `bsize₂` are different, therefore such is the iteration domain. We translate the
    # `i₁` index to its reciprocal `i₂` on the other side using the nD index.
    I₁ = position(bsize₁, i₁)

    # TODO: cleanup
    I₂x = ifelse(side₂ in sides_along(X_axis), ifelse(side₂ in first_sides(), 1, real_block_size(bsize₂)[1]), I₁[1])
    I₂y = ifelse(side₂ in sides_along(Y_axis), ifelse(side₂ in first_sides(), 1, real_block_size(bsize₂)[2]), I₁[2])
    I₂ = (I₂x, I₂y)
    # TODO: Unreadable but efficient and dimension-agnostic?
    # I₂ = ifelse.(
    #     in.(side₂, (sides_along(X_axis), sides_along(Y_axis))),
    #     ifelse.(side₂ in first_sides(), 1, real_block_size(bsize₂)),
    #     I₁
    # )

    i₂ = lin_position(bsize₂, I₂)

    ig₁ = i₁ + sg₁
    ig₂ = i₂ + sg₂

    vars_ghost_exchange(
        vars₁, i₁, ig₁, sg₁,
        vars₂, i₂, ig₂, sg₂,
        ghosts(bsize₁)
    )
end


function block_ghost_exchange(
    params::ArmonParameters, blk₁::LocalTaskBlock{V}, blk₂::LocalTaskBlock{V}, side::Side
) where {V}
    # Exchange between two blocks with (possibly) different dimensions, but the same length along `side`
    domain = border_domain(blk₁.size, side)
    block_ghost_exchange(params, domain,
        comm_vars(blk₁), blk₁.size,
        comm_vars(blk₂), blk₂.size,
        params.current_axis, side
    )
end


@generic_kernel function pack_to_array!(
    bsize::BlockSize, side::Side, array::V, vars::NTuple{N, V}
) where {N, V}
    idx = @index_2D_lin()
    itr = @iter_idx()

    (i, i_g) = divrem(itr - 1, ghosts(bsize))
    i_arr = (i_g * size_along(bsize, side) + i) * N

    # TODO: KernelAbstractions.Extras.@unroll ??
    for v in 1:N
        array[i_arr+v] = vars[v][idx]
    end
end


@generic_kernel function unpack_from_array!(
    bsize::BlockSize, side::Side, array::V, vars::NTuple{N, V}
) where {N, V}
    idx = @index_2D_lin()
    itr = @iter_idx()

    (i, i_g) = divrem(itr - 1, ghosts(bsize))
    i_arr = (i_g * size_along(bsize, side) + i) * N

    # TODO: KernelAbstractions.Extras.@unroll ??
    for v in 1:N
        array[i_arr+v] = vars[v][idx]
    end
end


function block_ghost_exchange(
    params::ArmonParameters, blk::LocalTaskBlock{V}, other_blk::RemoteTaskBlock{B}, side::Side
) where {V, B}
    if other_blk.rank == -1
        # `other_blk` is fake, this is the border of the global domain
        return boundary_conditions!(params, blk, side)
    end

    # Exchange between one local block and a remote block from another sub-domain
    # TODO: use RMA with processes local to the node.

    if V !== B
        # MPI buffers are not on the device. We first need to copy them to the host memory.
        device_to_host!(blk)
    end

    send_domain = border_domain(blk.size, side)
    recv_domain = shift_dir(send_domain, axis_of(side), side in first_sides() ? -ghosts(blk.size) : ghosts(blk.size))

    vars = comm_vars(blk)
    pack_to_array!(params, blk, send_domain, blk.size, side, other_blk.send_buf.data, vars)

    wait(params)  # Wait for the copy to complete

    MPI.Start(send_request)
    MPI.Start(recv_request)

    # Cooperative wait with Julia's scheduler
    wait(send_request)
    wait(recv_request)

    unpack_from_array!(params, blk, recv_domain, blk.size, side, other_blk.recv_buf.data, vars)

    if V !== B
        # MPI buffers are not on the device. Retreive the result of the exchange to the device.
        host_to_device!(blk)
    end
end


function block_ghost_exchange(params::ArmonParameters, grid::BlockGrid)
    side = first_side(params.current_axis)  # `Left`  or `Bottom`
    other_side = opposite_of(side)          # `Right` or `Top`
    other_side_offset = CartesianIndex(offset_to(other_side))

    @iter_blocks for blk in device_blocks(grid)
        # Parse through all blocks, each time updating the Left/Bottom neighbours
        other_blk = blk.neighbours[Int(side)]
        block_ghost_exchange(params, blk, other_blk, side)

        if !in_grid(blk.pos + other_side_offset, grid.grid_size)
            # Blocks at the edge of the grid along the current axis need an extra exchange
            other_blk = blk.neighbours[Int(other_side)]
            block_ghost_exchange(params, blk, other_blk, other_side)
        end
    end
end
