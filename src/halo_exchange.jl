
@generic_kernel function boundaryConditions!(
    stencil_width::Int, stride::Int, i_start::Int, d::Int, u_factor::T, v_factor::T,
    rho::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V, Emat::V
) where {T, V <: AbstractArray{T}}
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
        Emat[i] = Emat[i₊]

        i  -= d
        i₊ += d
    end
end


@generic_kernel function read_border_array!(
    side_length::Int, nghost::Int,
    rho::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V, Emat::V, value_array::V
) where V
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


@generic_kernel function write_border_array!(
    side_length::Int, nghost::Int,
    rho::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V, Emat::V, value_array::V
) where V
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


function read_border_array!(params::ArmonParameters, data::ArmonDualData, comm_array, side::Side)
    (; nx, ny) = params

    range = border_domain(params, side)
    side_length = (side == Left || side == Right) ? ny : nx

    return read_border_array!(params, device(data), range, side_length, comm_array)
end


function write_border_array!(params::ArmonParameters, data::ArmonDualData, comm_array, side::Side)
    (; nx, ny) = params

    range = ghost_domain(params, side)
    side_length = (side == Left || side == Right) ? ny : nx

    return write_border_array!(params, device(data), range, side_length, comm_array)
end


function copy_border_array_to_buffer!(params::ArmonParameters, data::ArmonDualData, side::Side)
    comm_array = get_send_comm_array(data, side)
    read_border_array!(params, data, comm_array, side)
    copy_to_send_buffer!(data, comm_array, side)
end


function copy_buffer_to_border_array!(params::ArmonParameters, data::ArmonDualData, side::Side)
    comm_array = get_recv_comm_array(data, side)
    copy_from_recv_buffer!(data, comm_array, side;)
    write_border_array!(params, data, comm_array, side)
end


function request_task(dependency, request)
    wait(dependency)
    MPI.Start(request)
    done = false
    while !done
        done = MPI.Test(request)
        yield()
    end
end


function boundaryConditions!(params::ArmonParameters{T}, data::ArmonDualData, side::Side, channel) where T
    if !has_neighbour(params, side)
        # No neighbour: global domain boundary conditions
        (u_factor::T, v_factor::T) = boundaryCondition(side, params.test)
        (i_start, loop_range, stride, d) = boundary_conditions_indexes(params, side)

        i_start -= stride  # Adjust for the fact that `@index_1D_lin()` is 1-indexed

        boundaryConditions!(params, device(data), loop_range, stride, i_start, d, u_factor, v_factor)
    else
        # Exchange with the neighbour the cells on the side
        copy_border_array_to_buffer!(params, data, side)

        send_request = data.requests[side].send
        recv_request = data.requests[side].recv

        if params.use_MPI && params.async_comms
            # Schedule the asynchronous exchange to start when the copy is done
            error("MPI async comms NYI")

            copy_done = Threads.Event()
            # TODO: notify(copy_done)  =>  in the copy task
            # TODO: wait for the copy to be done

            send_task = Threads.@spawn request_task(copy_done, send_request)
            recv_task = Threads.@spawn request_task(copy_done, recv_request)

            put!(channel, send_task)
            put!(channel, recv_task)

            copy_recv_task = Threads.@spawn begin
                wait(recv_task)
                # TODO: wait that the inner domain has been updated first (EOS)
                copy_buffer_to_border_array!(params, data, side)
            end

            put!(channel, copy_recv_task)
        else
            # Launch the communications and wait for their completion now in order to be able to 
            # measure them. 
            wait(params)

            @timeit params.timer "MPI" begin
                MPI.Start(send_request)
                MPI.Start(recv_request)

                MPI.Wait(send_request)
                MPI.Wait(recv_request)
            end

            copy_buffer_to_border_array!(params, data, side)
        end
    end
end


function boundaryConditions!(params::ArmonParameters, data::ArmonDualData, sides::Tuple{Vararg{Side}}, channel)
    # TODO : use active RMA instead? => maybe but it will (maybe) not work with GPUs: 
    #   https://www.open-mpi.org/faq/?category=runcuda
    # TODO : use CUDA/ROCM-aware MPI

    if !(params.use_MPI && params.async_comms)
        # We need a special ordering for synchronous communications in order to avoid deadlocks.
        # This method makes each successive sub-domain do their sides in reverse, therefore two
        # neighbouring sub-domains will attempt to exchange their cells at the same time:
        # ... (left then right) ; (right then left) ; (left then right) ; (right then left) ...
        if isodd(grid_coord_along(params))
            sides = reverse(sides)
        end
    end

    for side in sides
        boundaryConditions!(params, data, side, channel)
    end
end


function boundaryConditions!(params::ArmonParameters, data::ArmonDualData, sides::Symbol, channel)
    if sides === :outer_lb
        side = params.current_axis == X_axis ? Left : Bottom
        boundaryConditions!(params, data, (side,), channel)
    elseif sides === :outer_rt
        side = params.current_axis == X_axis ? Right : Top
        boundaryConditions!(params, data, (side,), channel)
    else
        error("Unknown sides: $sides")
    end
end


function boundaryConditions!(params::ArmonParameters, data::ArmonDualData)
    boundaryConditions!(params, data, sides_along(params.current_axis), nothing)
end
