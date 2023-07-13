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
    println(io, ": ROCm (block size: $(p.block_size))")
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
# Time step
#

@kernel function dtCFL_reduction(dx, dy, out, umat, vmat, cmat, domain_mask)
    Armon.@fast begin
        i = @index(Global)

        c = cmat[i]
        u = umat[i]
        v = vmat[i]
        mask = domain_mask[i]

        dt_x = dx / abs(max(abs(u + c), abs(u - c)) * mask)
        dt_y = dy / abs(max(abs(v + c), abs(v - c)) * mask)
        out[i] = min(dt_x, dt_y) 
    end
end


function Armon.dtCFL_kernel(params::ArmonParameters{<:Any, <:ROCBackend}, data::ArmonData, _, dx, dy)
    (; cmat, umat, vmat, domain_mask, work_array_1) = data

    # AMDGPU supports ArrayProgramming, but AMDGPU.mapreduce! is not as efficient as CUDA.mapreduce!
    # for large broadcasted arrays (it has something to do with allocations).
    # Therefore we first compute all time steps and store them in a work array to then reduce it.
    dtCFL_reduction_kernel = dtCFL_reduction(params.device, params.block_size)
    dtCFL_reduction_kernel(dx, dy, work_array_1, umat, vmat, cmat, domain_mask;
        ndrange=length(cmat))
    KernelAbstractions.synchronize(params.device)

    return reduce(min, work_array_1)
end

end
