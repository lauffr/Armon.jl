module ArmonCUDA

using Armon
isdefined(Base, :get_extension) ? (import CUDA) : (import ..CUDA)
using KernelAbstractions
import CUDA: CUDABackend


function Armon.init_device(::Val{:CUDA}, _)
    return CUDABackend()
end


Armon.device_array_type(::CUDABackend) = CUDA.CuArray


function Armon.print_device_info(io::IO, pad::Int, p::ArmonParameters{<:Any, <:CUDABackend})
    print_parameter(io, pad, "GPU", true, nl=false)
    println(io, ": CUDA (block size: $(p.block_size))")
end


function Armon.device_memory_info(::CUDABackend)
    free, total = CUDA.Mem.info()
    return (
        total = UInt64(total),
        free  = UInt64(free)
    )
end

end
