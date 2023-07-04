module ArmonAMDGPU

using Armon
isdefined(Base, :get_extension) ? (import AMDGPU) : (import ..AMDGPU)
using KernelAbstractions
using ROCKernels


function Armon.init_device(::Val{:ROCM}, _)
    AMDGPU.allowscalar(false)
    return ROCDevice()
end


Armon.device_array_type(::ROCDevice) = ROCArray


function print_device_info(io::IO, pad::Int, p::ArmonParameters{<:Any, <:ROCDevice})
    print_parameter(io, pad, "GPU", true, nl=false)
    println(io, ": ROCm (block size: $(p.block_size))")
end

#
# Custom reduction kernels
#

function dtCFL_kernel(::ArmonParameters{<:Any, <:ROCDevice}, ::ArmonData, range, dx, dy)
    (; cmat, umat, vmat, domain_mask, work_array_1) = data

    # TODO: test again in the newest versions
    # AMDGPU supports ArrayProgramming, but AMDGPU.mapreduce! is not as efficient as 
    # CUDA.mapreduce! for large broadcasted arrays. Therefore we first compute all time
    # steps and store them in a work array to then reduce it.
    gpu_dtCFL_reduction_euler! = gpu_dtCFL_reduction_euler_kernel!(params.device, params.block_size)
    gpu_dtCFL_reduction_euler!(dx, dy, work_array_1, umat, vmat, cmat, domain_mask;
        ndrange=length(cmat)) |> wait

    return reduce(min, work_array_1)
end

end
