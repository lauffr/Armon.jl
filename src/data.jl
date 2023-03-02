
"""
    ArmonData{V}

Generic array holder for all variables and temporary variables used throughout the solver.
`V` can be a `Vector` of floats (`Float32` or `Float64`) on CPU, `CuArray` or `ROCArray` on GPU.
`Vector`, `CuArray` and `ROCArray` are all subtypes of `AbstractArray`.
"""
struct ArmonData{V}
    x            :: V
    y            :: V
    rho          :: V
    umat         :: V
    vmat         :: V
    Emat         :: V
    pmat         :: V
    cmat         :: V
    gmat         :: V
    ustar        :: V
    pstar        :: V
    work_array_1 :: V
    work_array_2 :: V
    work_array_3 :: V
    work_array_4 :: V
    domain_mask  :: V
end


ArmonData(params::ArmonParameters{T}) where T = ArmonData(T, params.nbcell)
ArmonData(type::Type, size::Int64) = ArmonData(Vector{type}, size)

function ArmonData(array::Type{V}, size::Int64) where {V <: AbstractArray}
    return ArmonData{array}(
        array(undef, size),
        array(undef, size),
        array(undef, size),
        array(undef, size),
        array(undef, size),
        array(undef, size),
        array(undef, size),
        array(undef, size),
        array(undef, size),
        array(undef, size),
        array(undef, size),
        array(undef, size),
        array(undef, size),
        array(undef, size),
        array(undef, size),
        array(undef, size)
    )
end


main_variables() = (:x, :y, :rho, :umat, :vmat, :Emat, :pmat, :cmat, :gmat, :ustar, :pstar, :domain_mask)
main_variables(data::ArmonData) = getfield.(data, main_variables())

saved_variables() = (:x, :y, :rho, :umat, :vmat, :pmat)
saved_variables(data::ArmonData) = map(f -> getfield(data, f), saved_variables())


"""
    memory_required(params::ArmonParameters)

Compute the number of bytes needed on the device to allocate all data arrays

While the result is precise, it does not account for additional memory required by MPI buffers and
the solver.
"""
memory_required(params::ArmonParameters{T}) where T = memory_required_for(params.nbcell, T)

function memory_required(N, float_type)
    field_count = fieldcount(ArmonData{Vector{float_type}})
    floats = field_count * N
    return floats * sizeof(float_type)
end


"""
    ArmonDualData{DeviceArray, HostArray}

Holds two version of `ArmonData`, one for the device and one for the host, as well as the buffers
necessary for the halo exchange.

If the host and device are the same, the `device` and `host` fields point to the same data.
"""
struct ArmonDualData{DeviceArray <: AbstractArray, HostArray <: AbstractArray}
    device       :: GenericDevice
    device_data  :: ArmonData{DeviceArray}
    host_data    :: ArmonData{HostArray}
    comm_buffers :: Dict{Side, NamedTuple{(:send, :recv), NTuple{2, MPI.Buffer{HostArray}}}}
    requests     :: Dict{Side, NamedTuple{(:send, :recv), NTuple{2, MPI.AbstractRequest}}}
end


function ArmonDualData(params::ArmonParameters{T}) where T
    device_array = get_device_array(params){T}
    host_array = get_host_array(params){T}

    device_data = ArmonData(device_array, params.nbcell)
    if host_array == device_array
        host_data = device_data
    else
        host_data = ArmonData(host_array, params.nbcell)
    end

    # In case we don't use MPI since there is no neighbours no array is allocated
    comm_buffers = Dict{Side, NamedTuple{(:send, :recv), NTuple{2, MPI.Buffer{host_array}}}}()
    requests = Dict{Side, NamedTuple{(:send, :recv), NTuple{2, MPI.AbstractRequest}}}()
    for side in instances(Side)
        has_neighbour(params, side) || continue
        neighbour = neighbour_at(params, side)
        comm_buffers[side] = (
            send = MPI.Buffer(host_array(undef, params.comm_array_size)),
            recv = MPI.Buffer(host_array(undef, params.comm_array_size))
        )
        requests[side] = (
            send = MPI.Send_init(comm_buffers[side].send, params.cart_comm; dest=neighbour),
            recv = MPI.Recv_init(comm_buffers[side].recv, params.cart_comm; source=neighbour)
        )
    end

    return ArmonDualData{device_array, host_array}(params.device, device_data, host_data, comm_buffers, requests)
end


device_type(data::ArmonDualData) = data.device
device(data::ArmonDualData) = data.device_data
host(data::ArmonDualData) = data.host_data

iter_send_requests(data::ArmonDualData) = 
    Iterators.map(p -> first(p) => first(last(p)), 
        Iterators.filter(!MPI.isnull ∘ first ∘ last, data.requests))

iter_recv_requests(data::ArmonDualData) = 
    Iterators.map(p -> first(p) => last(last(p)), 
        Iterators.filter(!MPI.isnull ∘ last ∘ last, data.requests))

send_buffer(data::ArmonDualData, side::Side) = data.comm_buffers[side].send
recv_buffer(data::ArmonDualData, side::Side) = data.comm_buffers[side].recv

get_send_comm_array(data::ArmonDualData{H, H}, side::Side) where H = send_buffer(data, side).data
get_recv_comm_array(data::ArmonDualData{H, H}, side::Side) where H = recv_buffer(data, side).data

function get_send_comm_array(data::ArmonDualData{D, H}, side::Side) where {D, H}
    if side == Left
        comm_array = device(data).work_array_1
    elseif side == Right
        comm_array = device(data).work_array_2
    elseif side == Bottom
        comm_array = device(data).work_array_3
    else
        comm_array = device(data).work_array_4
    end
    return view(comm_array, Base.OneTo(length(send_buffer(data, side).data)))
end

get_recv_comm_array(data::ArmonDualData{D, H}, side::Side) where {D, H} = get_send_comm_array(data, side)


"""
    device_to_host!(data::ArmonDualData)

Copies all `main_variables()` from the device to the host data. A no-op if the host is the device.
"""
device_to_host!(::ArmonDualData{D, D}) where D = nothing


"""
    host_to_device!(data::ArmonDualData)

Copies all `main_variables()` from the host to the device data. A no-op if the host is the device.
"""
host_to_device!(::ArmonDualData{D, D}) where D = nothing


function device_to_host!(data::ArmonDualData{D, H}) where {D, H}
    for var in main_variables()
        copyto!(getfield(host(data), var), getfield(device(data), var))
    end
end


function host_to_device!(data::ArmonDualData{D, H}) where {D, H}
    for var in main_variables()
        copyto!(getfield(device(data), var), getfield(host(data), var))
    end
end


# In homogenous configurations, the data is already in the send buffer because of `get_send_comm_array` and `get_recv_comm_array`
copy_to_send_buffer!(::ArmonDualData{H, H}, ::H, ::Side; dependencies=NoneEvent()) where H = dependencies
copy_to_send_buffer!(::ArmonDualData{H, H}, ::H, ::H; dependencies=NoneEvent()) where H = dependencies
copy_from_recv_buffer!(::ArmonDualData{H, H}, ::H, ::Side; dependencies=NoneEvent()) where H = dependencies
copy_from_recv_buffer!(::ArmonDualData{H, H}, ::H, ::H; dependencies=NoneEvent()) where H = dependencies


function copy_to_send_buffer!(data::ArmonDualData{D, H}, array::D, side::Side; 
        dependencies=NoneEvent()) where {D, H}
    buffer_data = send_buffer(data, side).data
    copy_to_send_buffer!(data, array, buffer_data; dependencies)
end


function copy_to_send_buffer!(data::ArmonDualData{D, H}, array::D, buffer::H;
        dependencies=NoneEvent()) where {D, H}
    array_data = view(array, 1:length(buffer))
    wait(dependencies)  # We cannot wait for CPU events on the GPU
    return async_copy!(device_type(data), buffer, array_data)
end


function copy_from_recv_buffer!(data::ArmonDualData{D, H}, array::D, side::Side;
        dependencies=NoneEvent()) where {D, H}
    buffer_data = recv_buffer(data, side).data
    copy_from_recv_buffer!(data, array, buffer_data; dependencies)
end


function copy_from_recv_buffer!(data::ArmonDualData{D, H}, array::D, buffer::H;
        dependencies=NoneEvent()) where {D, H}
    array_data = view(array, 1:length(buffer))
    wait(dependencies)  # We cannot wait for CPU events on the GPU
    return async_copy!(device_type(data), array_data, buffer)
end


ArmonDataOrDual = Union{ArmonData, ArmonDualData}
