
function limiter_type_to_int(params::ArmonParameters)::Cint
    if     params.riemann_limiter isa NoLimiter       return 0
    elseif params.riemann_limiter isa MinmodLimiter   return 1
    elseif params.riemann_limiter isa SuperbeeLimiter return 2
    else
        error("This limiter is not recognized by armon_cpp")
    end
end


function test_case_to_int(params::ArmonParameters)::Cint
    if     params.test isa Sod       return 0
    elseif params.test isa Sod_y     return 1
    elseif params.test isa Sod_circ  return 2
    elseif params.test isa Bizarrium return 3
    elseif params.test isa Sedov     return 4
    else
        error("This test case is not recognized by armon_cpp")
    end
end


function get_init_test_params(params::ArmonParameters, test_params_ptr::Ptr{Nothing}, len::Cint)
    test_params_ptr = Ptr{data_type(params)}(test_params_ptr)

    test_params = init_test_params(params.test)
    length(test_params) > len && error("the given array is not big enough, expected $(length(test_params)), got: $len")

    for (i, p) in enumerate(test_params[2:end])
        unsafe_store!(test_params_ptr, p, i)
    end

    if params.test isa Sedov
        unsafe_store!(test_params_ptr, params.test.r, length(test_params))
    end

    return nothing
end


raise_cpp_exception(str::Cstring) = error("C++ exception: " * Base.unsafe_string(str))


function init_armon_cpp(params::ArmonParameters{T}) where T
    params_t = typeof(params)

    names = map(fieldnames(params_t)) do name
        ccall(:jl_symbol_name, Ptr{Nothing}, (Symbol,), name)
    end |> collect

    offsets = fieldoffset.(params_t, 1:fieldcount(params_t)) |> collect

    is_error = ccall(Kokkos.get_symbol(params.kokkos_lib, :init_params_offsets),
        Cchar, (Ptr{Nothing}, Ptr{Nothing}, Cint),
        pointer(names), pointer(offsets), fieldcount(params_t)
    )

    if is_error != 0
        error("missing fields for armon_cpp initialization.")
    end

    ccall(Kokkos.get_symbol(params.kokkos_lib, :init_callbacks),
        Cchar, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
        @cfunction(limiter_type_to_int, Cint, (Ref{ArmonParameters},)),
        @cfunction(test_case_to_int, Cint, (Ref{ArmonParameters},)),
        @cfunction(get_init_test_params, Cvoid, (Ref{ArmonParameters}, Ptr{Nothing}, Cint)),
        @cfunction(raise_cpp_exception, Cvoid, (Cstring,))
    )

    armon_cpp_eltype_size = ccall(Kokkos.get_symbol(params.kokkos_lib, :data_type_size),
        Cint, ()
    )

    if armon_cpp_eltype_size != sizeof(T)
        error("eltype size mismatch: expected $(sizeof(T)) (for $T), got $armon_cpp_eltype_size")
    end
end


@generated function kokkos_dtCFL(
    params::ArmonParameters{T}, data::ArmonData{V}, range
) where {T, V <: AbstractArray{T}}
    quote
        return ccall(Kokkos.get_symbol(params.kokkos_lib, :dt_CFL),
            T, (Ptr{Nothing}, Int64, Int64, Int64,
                Ref{V}, Ref{V}, Ref{V}, Ref{V}),
            pointer_from_objref(params), first(range), step(range), last(range),
            data.umat, data.vmat, data.cmat, data.domain_mask
        )
    end
end


@generated function kokkos_conservation_vars(
    params::ArmonParameters{T}, data::ArmonData{V}, range
) where {T, V <: AbstractArray{T}}
    quote
        total_mass_ref = Ref{T}(0)
        total_energy_ref = Ref{T}(0)
        ccall(Kokkos.get_symbol(params.kokkos_lib, :conservation_vars),
            Cvoid, (Ptr{Nothing}, Int64, Int64, Int64,
                    Ref{V}, Ref{V}, Ref{V}, Ref{T}, Ref{T}),
            pointer_from_objref(params), first(range), step(range), last(range),
            data.rho, data.Emat, data.domain_mask,
            total_mass_ref, total_energy_ref
        )
        return total_mass_ref[], total_energy_ref[]
    end
end
