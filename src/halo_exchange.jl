

@generic_kernel function boundaryConditions!(stencil_width::Int, stride::Int, i_start::Int, d::Int,
        u_factor::T, v_factor::T, rho::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, async, label=boundaryConditions!)

    idx = @index_1D_lin()
    i  = idx * stride + i_start
    i₊ = i + d

    for _ in 1:stencil_width
        rho[i]  = rho[i₊]
        umat[i] = umat[i₊] * u_factor
        vmat[i] = vmat[i₊] * v_factor
        pmat[i] = pmat[i₊]
        cmat[i] = cmat[i₊]
        gmat[i] = gmat[i₊]

        i  -= d
        i₊ += d
    end
end


@generic_kernel function read_border_array!(side_length::Int, nghost::Int,
        rho::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V, Emat::V, value_array::V) where V
    @kernel_options(add_time, async, label=border_array)

    idx = @index_2D_lin()
    itr = @iter_idx()

    (i, i_g) = divrem(itr - 1, nghost)
    i_arr = (i_g * side_length + i) * 7

    value_array[i_arr+1] =  rho[idx]
    value_array[i_arr+2] = umat[idx]
    value_array[i_arr+3] = vmat[idx]
    value_array[i_arr+4] = pmat[idx]
    value_array[i_arr+5] = cmat[idx]
    value_array[i_arr+6] = gmat[idx]
    value_array[i_arr+7] = Emat[idx]
end


@generic_kernel function write_border_array!(side_length::Int, nghost::Int,
        rho::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V, Emat::V, value_array::V) where V
    @kernel_options(add_time, async, label=border_array)

    idx = @index_2D_lin()
    itr = @iter_idx()

    (i, i_g) = divrem(itr - 1, nghost)
    i_arr = (i_g * side_length + i) * 7

     rho[idx] = value_array[i_arr+1]
    umat[idx] = value_array[i_arr+2]
    vmat[idx] = value_array[i_arr+3]
    pmat[idx] = value_array[i_arr+4]
    cmat[idx] = value_array[i_arr+5]
    gmat[idx] = value_array[i_arr+6]
    Emat[idx] = value_array[i_arr+7]
end



function read_border_array!(params::ArmonParameters{T}, data::ArmonData{V}, value_array::W, side::Symbol;
        dependencies=NoneEvent(), no_threading=false) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; nghost, nx, ny, row_length) = params
    (; tmp_comm_array) = data
    @indexing_vars(params)

    if side == :left
        main_range = @i(1, 1):row_length:@i(1, ny)
        inner_range = 1:nghost
        side_length = ny
    elseif side == :right
        main_range = @i(nx-nghost+1, 1):row_length:@i(nx-nghost+1, ny)
        inner_range = 1:nghost
        side_length = ny
    elseif side == :top
        main_range = @i(1, ny-nghost+1):row_length:@i(1, ny)
        inner_range = 1:nx
        side_length = nx
    elseif side == :bottom
        main_range = @i(1, 1):row_length:@i(1, nghost)
        inner_range = 1:nx
        side_length = nx
    else
        error("Unknown side: $side")
    end

    range = DomainRange(main_range, inner_range)
    event = read_border_array!(params, data, range, side_length, tmp_comm_array;
        dependencies, no_threading)

    if params.use_gpu
        # Copy `tmp_comm_array` from the GPU to the CPU in `value_array`
        event = async_copy!(params.device, value_array, tmp_comm_array; dependencies=event)
        event = @time_event_a "border_array" event
    end

    return event
end


function write_border_array!(params::ArmonParameters{T}, data::ArmonData{V}, value_array::W, side::Symbol;
        dependencies=NoneEvent(), no_threading=false) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; nghost, nx, ny, row_length) = params
    (; tmp_comm_array) = data
    @indexing_vars(params)

    # Write the border array to the ghost cells of the data arrays
    if side == :left
        main_range = @i(1-nghost, 1):row_length:@i(1-nghost, ny)
        inner_range = 1:nghost
        side_length = ny
    elseif side == :right
        main_range = @i(nx+1, 1):row_length:@i(nx+1, ny)
        inner_range = 1:nghost
        side_length = ny
    elseif side == :top
        main_range = @i(1, ny+1):row_length:@i(1, ny+nghost)
        inner_range = 1:nx
        side_length = nx
    elseif side == :bottom
        main_range = @i(1, 1-nghost):row_length:@i(1, 0)
        inner_range = 1:nx
        side_length = nx
    else
        error("Unknown side: $side")
    end

    if params.use_gpu
        # Copy `value_array` from the CPU to the GPU in `tmp_comm_array`
        event = async_copy!(params.device, tmp_comm_array, value_array; dependencies)
        event = @time_event_a "border_array" event
    else
        event = dependencies
    end

    range = DomainRange(main_range, inner_range)
    event = write_border_array!(params, data, range, side_length, tmp_comm_array; 
        dependencies=event, no_threading)

    return event
end


function exchange_with_neighbour(params::ArmonParameters{T}, array::V, neighbour_rank::Int) where {T, V <: AbstractArray{T}}
    @perf_task "comms" "MPI_sendrecv" @time_expr_a "boundaryConditions!_MPI" MPI.Sendrecv!(array, 
        neighbour_rank, 0, array, neighbour_rank, 0, params.cart_comm)
end


function boundaryConditions!(params::ArmonParameters{T}, data::ArmonData{V}, host_array::W, axis::Axis; 
        dependencies=NoneEvent(), no_threading=false) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; neighbours, cart_coords) = params
    # TODO : use active RMA instead? => maybe but it will (maybe) not work with GPUs: 
    #   https://www.open-mpi.org/faq/?category=runcuda
    # TODO : use CUDA/ROCM-aware MPI
    # TODO : use 4 views for each side for each variable ? (2 will be contiguous, 2 won't)
    #   <- pre-calculate them!
    # TODO : try to mix the comms: send to left and receive from right, then vice-versa. 
    #  Maybe it can speed things up?    

    # We only exchange the ghost domains along the current axis.
    # even x/y coordinate in the cartesian process grid:
    #   - send+receive left  or top
    #   - send+receive right or bottom
    # odd  x/y coordinate in the cartesian process grid:
    #   - send+receive right or bottom
    #   - send+receive left  or top
    (cx, cy) = cart_coords
    if axis == X_axis
        if cx % 2 == 0
            order = [:left, :right]
        else
            order = [:right, :left]
        end
    else
        if cy % 2 == 0
            order = [:top, :bottom]
        else
            order = [:bottom, :top]
        end
    end

    comm_array = params.use_gpu ? host_array : data.tmp_comm_array

    prev_event = dependencies

    for side in order
        neighbour = neighbours[side]
        if neighbour == MPI.PROC_NULL
            prev_event = boundaryConditions!(params, data, side;
                dependencies=prev_event, no_threading)
        else
            read_event = read_border_array!(params, data, comm_array, side; 
                dependencies=prev_event, no_threading)
            Event(exchange_with_neighbour, params, comm_array, neighbour; 
                dependencies=read_event) |> wait
            prev_event = write_border_array!(params, data, comm_array, side; 
                no_threading)
        end
    end

    return prev_event
end
