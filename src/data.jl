
"""
    ArmonData{V}

Generic array holder for all variables and temporary variables used throughout the solver.
`V` can be a `Vector` of floats (`Float32` or `Float64`) on CPU, `CuArray` or `ROCArray` on GPU.
`Vector`, `CuArray` and `ROCArray` are all subtypes of `AbstractArray`.
"""
struct ArmonData{V}
    x::V
    y::V
    rho::V
    umat::V
    vmat::V
    Emat::V
    pmat::V
    cmat::V
    gmat::V
    ustar::V
    pstar::V
    work_array_1::V
    work_array_2::V
    work_array_3::V
    work_array_4::V
    domain_mask::V
    tmp_comm_array::V
end


function ArmonData(params::ArmonParameters{T}) where T
    return ArmonData(T, params.nbcell, params.comm_array_size)
end


function ArmonData(type::Type, size::Int64, tmp_comm_size::Int64)
    return ArmonData{Vector{type}}(
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, tmp_comm_size)
    )
end


function data_to_gpu(data::ArmonData{V}, device_array) where {T, V <: AbstractArray{T}}
    return ArmonData{device_array{T}}(
        device_array(data.x),
        device_array(data.y),
        device_array(data.rho),
        device_array(data.umat),
        device_array(data.vmat),
        device_array(data.Emat),
        device_array(data.pmat),
        device_array(data.cmat),
        device_array(data.gmat),
        device_array(data.ustar),
        device_array(data.pstar),
        device_array(data.work_array_1),
        device_array(data.work_array_2),
        device_array(data.work_array_3),
        device_array(data.work_array_4),
        device_array(data.domain_mask),
        device_array(data.tmp_comm_array)
    )
end


function data_from_gpu(host_data::ArmonData{V}, device_data::ArmonData{W}) where 
        {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    # We only need to copy the non-temporary arrays 
    copyto!(host_data.x, device_data.x)
    copyto!(host_data.y, device_data.y)
    copyto!(host_data.rho, device_data.rho)
    copyto!(host_data.umat, device_data.umat)
    copyto!(host_data.vmat, device_data.vmat)
    copyto!(host_data.Emat, device_data.Emat)
    copyto!(host_data.pmat, device_data.pmat)
    copyto!(host_data.cmat, device_data.cmat)
    copyto!(host_data.gmat, device_data.gmat)
    copyto!(host_data.ustar, device_data.ustar)
    copyto!(host_data.pstar, device_data.pstar)
end


function memory_required_for(params::ArmonParameters{T}) where T
    return memory_required_for(params.nbcell, params.comm_array_size, T)
end


function memory_required_for(N, communication_array_size, float_type)
    field_count = fieldcount(ArmonData{AbstractArray{float_type}})
    floats = (field_count - 1) * N + communication_array_size
    return floats * sizeof(float_type)
end
