module ArmonAMDGPU

using Armon
import Armon: ArmonData
isdefined(Base, :get_extension) ? (import AMDGPU) : (import ..AMDGPU)
using KernelAbstractions
import AMDGPU: ROCBackend


function Armon.init_device(::Val{:ROCM}, _)
    return ROCBackend()
end


Armon.device_array_type(::ROCBackend) = AMDGPU.ROCArray


function Armon.print_device_info(io::IO, pad::Int, p::ArmonParameters{<:Any, <:ROCBackend})
    Armon.print_parameter(io, pad, "GPU", true, nl=false)
    println(io, ": ROCm (block size: ", join(p.block_size, '×'), ")")
end


function Armon.device_memory_info(::ROCBackend)
    @static if pkgversion(AMDGPU) ≥ v"0.5"
        free, total = AMDGPU.Runtime.Mem.info()
    else
        free_p = Ref{UInt64}()
        total_p = Ref{UInt64}()
        ccall((:hipMemGetInfo, AMDGPU.libhip),
            AMDGPU.HIP.hipError_t, (Ptr{Csize_t}, Ptr{Csize_t}),
            free_p, total_p)
        free = free_p[]
        total = total_p[]
    end

    return (
        total = UInt64(total),
        free  = UInt64(free)
    )
end


# TODO: profiling with rocprofile with roctracer library


#
# Reductions
#

@kernel function dtCFL_reduction(dx, dy, out, umat, vmat, cmat, domain_mask)
    Armon.@fast begin
        i = @index(Global)
        out[i] = Armon.dtCFL_kernel_reduction(umat[i], vmat[i], cmat[i], domain_mask[i], dx, dy)
    end
end


function Armon.dtCFL_kernel(params::ArmonParameters{<:Any, <:ROCBackend}, data::ArmonData, _, dx, dy)
    (; cmat, umat, vmat, domain_mask, work_array_1) = data

    # AMDGPU supports ArrayProgramming, but AMDGPU.mapreduce! is not as efficient as CUDA.mapreduce!
    # for large broadcasted arrays (it has something to do with allocations).
    # Therefore we first compute all time steps and store them in a work array to then reduce it.
    dtCFL_reduction_kernel = dtCFL_reduction(params.device, params.block_size)
    dtCFL_reduction_kernel(dx, dy, work_array_1, umat, vmat, cmat, domain_mask;
        ndrange=(length(cmat), 1, 1))
    KernelAbstractions.synchronize(params.device)

    return reduce(min, work_array_1)
end


@kernel function conservation_vars_reduction(out_mass, out_energy, rho, Emat, domain_mask)
    Armon.@fast begin
        i = @index(Global)
        out_mass[i], out_energy[i] = Armon.conservation_vars_kernel_reduction(rho[i], Emat[i], domain_mask[i])
    end
end


id2(a, b) = (a, b)


function Armon.conservation_vars_kernel(params::ArmonParameters{T, <:ROCBackend}, data::ArmonData, _) where T
    (; rho, Emat, domain_mask, work_array_1, work_array_2) = data

    conservation_reduction_kernel = conservation_vars_reduction(params.device, params.block_size)
    conservation_reduction_kernel(work_array_1, work_array_2, rho, Emat, domain_mask;
        ndrange=(length(rho), 1, 1))
    KernelAbstractions.synchronize(params.device)

    return mapreduce(id2, .+, work_array_1, work_array_2; init=(zero(T), zero(T)))
end

end
