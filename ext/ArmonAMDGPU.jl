module ArmonAMDGPU

using Armon
import Armon: ArmonData
isdefined(Base, :get_extension) ? (import AMDGPU) : (import ..AMDGPU)
using KernelAbstractions
import AMDGPU: ROCBackend


function Armon.init_device(::Val{:ROCM}, _)
    return AMDGPU.default_device()
end


Armon.device_array_type(::ROCBackend) = AMDGPU.ROCArray


function Armon.print_device_info(io::IO, pad::Int, p::ArmonParameters{<:Any, <:ROCBackend})
    Armon.print_parameter(io, pad, "GPU", true, nl=false)
    println(io, ": ROCm (block size: $(p.block_size))")
end


function Armon.device_memory_info(::ROCBackend)
    free, total = AMDGPU.Runtime.Mem.info()
    return (
        total = UInt64(total),
        free  = UInt64(free)
    )
end

end
