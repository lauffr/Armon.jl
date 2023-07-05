module ArmonKokkos

using Armon
import Armon: ArmonData, Side
import Armon: NoLimiter, MinmodLimiter, SuperbeeLimiter
import Armon: Sod, Sod_y, Sod_circ, Bizarrium, Sedov

isdefined(Base, :get_extension) ? (import Kokkos) : (import ..Kokkos)
import Kokkos: CMakeKokkosProject, option!
using MPI


function Armon.init_device(::Val{:Kokkos}, _)
    !Kokkos.is_initialized() && error("Kokkos should be initialized before creating ArmonParameters")
    return Base.invokelatest(Kokkos.DEFAULT_DEVICE_SPACE)
end


function Armon.init_backend(::Val{:Kokkos},
    flt_type, cmake_options, kokkos_options, use_MPI, is_root, global_comm
)
    !Kokkos.is_initialized() && error("Kokkos has not yet been initialized")

    armon_cpp_lib_src = joinpath(@__DIR__, "..", "lib", "armon_cpp")
    armon_cpp_lib_build = joinpath(Kokkos.KOKKOS_BUILD_DIR, "armon_cpp")
    armon_cpp = CMakeKokkosProject(armon_cpp_lib_src, "libarmon_cpp";
        build_dir=armon_cpp_lib_build, cmake_options, kokkos_options)
    option!(armon_cpp, "USE_SINGLE_PRECISION", flt_type == Float32; prefix="")

    if use_MPI
        is_root && Kokkos.compile(armon_cpp)
        MPI.Barrier(global_comm)
    else
        Kokkos.compile(armon_cpp)
    end
    kokkos_lib = Kokkos.load_lib(armon_cpp)

    return armon_cpp, kokkos_lib
end


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
    test_params_ptr = Ptr{Armon.data_type(params)}(test_params_ptr)

    test_params = Armon.init_test_params(params.test)
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


function Armon.post_init_device(::Val{:Kokkos}, params::ArmonParameters{T}) where T
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


function Armon.print_device_info(io::IO, pad::Int, p::ArmonParameters{<:Any, <:Kokkos.ExecutionSpace})
    Armon.print_parameter(io, pad, "use_kokkos", true)
    Armon.print_parameter(io, pad, "device", nameof(Kokkos.main_space_type(p.device)))
    Armon.print_parameter(io, pad, "memory", nameof(Kokkos.main_space_type(Kokkos.memory_space(p.device))))
end


function Armon.device_memory_info(exec::Kokkos.ExecutionSpace)
    if exec isa Kokkos.Cuda || exec isa Kokkos.HIP
        free, total = Kokkos.BackendFunctions.memory_info()
        return (
            total = UInt64(total),
            free  = UInt64(free)
        )
    elseif exec isa Kokkos.Serial || exec isa Kokkos.OpenMP
        return Armon.device_memory_info(CPU_HP())
    else
        error("`device_memory_info` for $(Kokkos.main_space_type(exec)) NYI")
    end
end


function Base.wait(::ArmonParameters{<:Any, <:Kokkos.ExecutionSpace}, _)
    Kokkos.fence()
end

#
# Array allocation
#

function Armon.host_array_type(::Kokkos.ExecutionSpace)
    return Kokkos.View{T, D,
        Kokkos.array_layout(Kokkos.DEFAULT_HOST_SPACE),
        Kokkos.DEFAULT_HOST_MEM_SPACE
    } where {T, D}
end


function Armon.device_array_type(device::Kokkos.ExecutionSpace)
    return Kokkos.View{T, D,
        Kokkos.array_layout(device),
        Kokkos.memory_space(device)
    } where {T, D}
end

#
# Custom reduction kernels
#

@generated function Armon.dtCFL_kernel(
    params::ArmonParameters{T, <:Kokkos.ExecutionSpace}, data::ArmonData{V}, range, dx, dy
) where {T, V <: AbstractArray{T}}
    quote
        return ccall(Kokkos.get_symbol(params.kokkos_lib, :dt_CFL),
            T, (Ptr{Nothing}, Int64, Int64, Int64,
                Ref{V}, Ref{V}, Ref{V}, Ref{V},
                T, T),
            pointer_from_objref(params), first(range), step(range), last(range),
            data.umat, data.vmat, data.cmat, data.domain_mask,
            dx, dy
        )
    end
end


@generated function Armon.conservation_vars_kernel(
    params::ArmonParameters{T, <:Kokkos.ExecutionSpace}, data::ArmonData{V}, range
) where {T, V <: Kokkos.View{T}}
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

#
# Copies
#

function Armon.copy_to_send_buffer!(::ArmonDualData{D, H, <:Kokkos.ExecutionSpace}, array::D, buffer::H;
    dependencies=NoneEvent()
) where {D, H}
    wait(dependencies)
    array_data = Kokkos.subview(array, 1:length(buffer))
    Kokkos.deep_copy(buffer, array_data)  # TODO: change to asynchronous deep copy
    return NoneEvent()
end


function Armon.copy_from_recv_buffer!(::ArmonDualData{D, H, <:Kokkos.ExecutionSpace}, array::D, buffer::H;
    dependencies=NoneEvent()
) where {D, H}
    wait(dependencies)
    array_data = Kokkos.subview(array, 1:length(buffer))
    Kokkos.deep_copy(array_data, buffer)  # TODO: change to asynchronous deep copy
    return NoneEvent()
end


function Armon.get_send_comm_array(data::ArmonDualData{D, H, <:Kokkos.ExecutionSpace}, side::Side) where {D, H}
    # Kokkos functions (like deep_copy, etc...) called on the comm array require an actual
    # Kokkos.View, not a view.
    array_view = Base.invoke(Armon.get_send_comm_array, Tuple{ArmonDualData{D, H, Any}}, data, side)
    return Kokkos.subview(parent(array_view), parentindices(array_view))
end

end
