module ArmonCUDA

using Armon
isdefined(Base, :get_extension) ? (import CUDA) : (import ..CUDA)
using KernelAbstractions
import CUDA: CUDABackend


Armon.create_device(::Val{:CUDA}) = CUDABackend()
Armon.device_array_type(::CUDABackend) = CUDA.CuArray


function Armon.print_device_info(io::IO, pad::Int, p::ArmonParameters{<:Any, <:CUDABackend})
    Armon.print_parameter(io, pad, "GPU", true, nl=false)
    println(io, ": CUDA (block size: ", join(p.block_size, '×'), ")")
end


function Armon.device_memory_info(::CUDABackend)
    free, total = CUDA.Mem.info()
    return (
        total = UInt64(total),
        free  = UInt64(free)
    )
end


function cuda_kernel_start(_, _)
    # Equivalent to CUDA.@profile
    CUDA.Profile.start()
    return nothing
end


function cuda_kernel_end(_, _, _)
    CUDA.Profile.stop()
end


function __init__()
    Armon.register_kernel_callback(Armon.KernelCallback((
        :CUDA_kernels,
        cuda_kernel_start,
        cuda_kernel_end
    )))    
end

end
