
# TODO: remove `async` from @kernel_options and `no_threading` kwarg

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



function read_border_array!(params::ArmonParameters, data::ArmonDualData, comm_array, side::Side;
        dependencies=NoneEvent(), no_threading=false)
    (; nx, ny) = params

    range = border_domain(params, side)
    side_length = (side == Left || side == Right) ? ny : nx

    return read_border_array!(params, device(data), range, side_length, comm_array; dependencies, no_threading)
end


function write_border_array!(params::ArmonParameters, data::ArmonDualData, comm_array, side::Side;
        dependencies=NoneEvent(), no_threading=false)
    (; nx, ny) = params

    range = ghost_domain(params, side)
    side_length = (side == Left || side == Right) ? ny : nx

    return write_border_array!(params, device(data), range, side_length, comm_array; dependencies, no_threading)
end


function boundaryConditions!(params::ArmonParameters{T}, data::ArmonDualData, side::Side;
        dependencies=NoneEvent(), no_threading=false) where T
    if !has_neighbour(params, side)
        # No neighbour: global domain boundary conditions
        (u_factor::T, v_factor::T) = boundaryCondition(side, params.test)
        (i_start, loop_range, stride, d) = boundary_conditions_indexes(params, side)

        i_start -= stride  # Adjust for the fact that `@index_1D_lin()` is 1-indexed

        return boundaryConditions!(params, device(data), loop_range, stride, i_start, d, 
            u_factor, v_factor; dependencies, no_threading)
    else
        comm_array = device(data).work_array_1

        # Exchange with the neighbour the cells on the side
        dependencies = read_border_array!(params, data, comm_array, side; dependencies)
        dependencies = copy_to_send_buffer!(data, comm_array, side; dependencies)

        # Schedule the exchange to start when the copy is done
        return Event(data.requests[side]; dependencies) do requests
            MPI.Start(requests.send)
            MPI.Start(requests.recv)
        end
    end
end


function boundaryConditions!(params::ArmonParameters, data::ArmonDualData, sides::Tuple{Vararg{Side}};
        dependencies=NoneEvent())
    # TODO : use active RMA instead? => maybe but it will (maybe) not work with GPUs: 
    #   https://www.open-mpi.org/faq/?category=runcuda
    # TODO : use CUDA/ROCM-aware MPI

    events = Event[]
    for side in sides
        push!(events, boundaryConditions!(params, data, side; dependencies))
    end
    dependencies = MultiEvent(tuple(events...))

    if params.use_MPI && !params.async_comms
        return post_boundary_conditions(params, data; dependencies)
    end

    return dependencies
end


function boundaryConditions!(params::ArmonParameters, data::ArmonDualData, sides::Symbol; dependencies=NoneEvent())
    if sides === :outer_lb
        side = params.current_axis == X_axis ? Left : Bottom
        boundaryConditions!(params, data, (side,); dependencies)
    elseif sides === :outer_rt
        side = params.current_axis == X_axis ? Right : Top
        boundaryConditions!(params, data, (side,); dependencies)
    else
        error("Unknown sides: $sides")
    end
end


function boundaryConditions!(params::ArmonParameters, data::ArmonDualData; dependencies=NoneEvent())
    boundaryConditions!(params, data, sides_along(params.current_axis); dependencies)
end


function post_boundary_conditions(params::ArmonParameters, data::ArmonDualData;
        dependencies=NoneEvent())
    !params.use_MPI && return dependencies

    # Parallelize each wait and the work after each request completion

    send_events = map(iter_send_requests(data)) do (_, send_request)
        Event(MPI.Wait, send_request; dependencies)
    end

    recv_events = map(iter_recv_requests(data)) do (side, recv_request)
        recv_deps = Event(MPI.Wait, recv_request; dependencies)
        comm_array = device(data).work_array_1
        recv_deps = copy_from_recv_buffer!(data, comm_array, side; dependencies=recv_deps)
        recv_deps = write_border_array!(params, data, comm_array, side; dependencies=recv_deps)
        return recv_deps
    end

    return MultiEvent((send_events..., recv_events...))
end
