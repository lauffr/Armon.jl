module ArmonCUDA

using Armon
isdefined(Base, :get_extension) ? (import CUDA) : (import ..CUDA)
using KernelAbstractions
using CUDAKernels


function Armon.init_device(::Val{:CUDA}, _)
    return CUDADevice()
end


Armon.device_array_type(::CUDADevice) = CUDA.CuArray


function Armon.print_device_info(io::IO, pad::Int, p::ArmonParameters{<:Any, <:CUDADevice})
    Armon.print_parameter(io, pad, "GPU", true, nl=false)
    println(io, ": CUDA (block size: $(p.block_size))")
end


function Armon.device_memory_info(::CUDADevice)
    free, total = CUDA.Mem.info()
    return (
        total = UInt64(total),
        free  = UInt64(free)
    )
end

end
