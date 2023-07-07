module ArmonAMDGPU

using Armon
import Armon: ArmonData
isdefined(Base, :get_extension) ? (import AMDGPU) : (import ..AMDGPU)
using KernelAbstractions
using ROCKernels


function Armon.init_device(::Val{:ROCM}, _)
    return AMDGPU.default_device()
end


Armon.device_array_type(::ROCDevice) = AMDGPU.ROCArray


function Armon.print_device_info(io::IO, pad::Int, p::ArmonParameters{<:Any, <:ROCDevice})
    Armon.print_parameter(io, pad, "GPU", true, nl=false)
    println(io, ": ROCm (block size: $(p.block_size))")
end


function Armon.device_memory_info(::ROCDevice)
    free, total = AMDGPU.Runtime.Mem.info()
    return (
        total = UInt64(total),
        free  = UInt64(free)
    )
end

#
# Custom reduction kernels
#

function Armon.dtCFL_kernel(params::ArmonParameters{<:Any, <:ROCDevice}, data::ArmonData, _, dx, dy)
    (; cmat, umat, vmat, domain_mask, work_array_1) = data

    # TODO: test again in the newest versions
    # AMDGPU supports ArrayProgramming, but AMDGPU.mapreduce! is not as efficient as 
    # CUDA.mapreduce! for large broadcasted arrays. Therefore we first compute all time
    # steps and store them in a work array to then reduce it.
    gpu_dtCFL_reduction_euler! = Armon.gpu_dtCFL_reduction_euler_kernel!(params.device, params.block_size)
    gpu_dtCFL_reduction_euler!(dx, dy, work_array_1, umat, vmat, cmat, domain_mask;
        ndrange=length(cmat)) |> wait

    return reduce(min, work_array_1)
end


function Armon.conservation_vars_kernel(params::ArmonParameters{T, <:ROCDevice}, data::ArmonData, range) where T
    # TODO: shouldn't be needed in KA.jl v9
    (; rho, Emat, domain_mask) = data
    (; dx) = params

    ds = dx * dx
    total_mass = @inbounds reduce(+, @views (
        rho[range] .* domain_mask[range] .* ds))
    total_energy = @inbounds reduce(+, @views (
        rho[range] .* Emat[range] .* domain_mask[range] .* ds))

    return total_mass, total_energy
end

end
