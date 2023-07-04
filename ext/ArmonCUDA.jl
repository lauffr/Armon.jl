module ArmonCUDA

using Armon
isdefined(Base, :get_extension) ? (import CUDA) : (import ..CUDA)
using KernelAbstractions
using CUDAKernels


function Armon.init_device(::Val{:CUDA}, _)
    CUDA.allowscalar(false)
    return CUDADevice()
end


Armon.device_array_type(::CUDADevice) = CuArray


function print_device_info(io::IO, pad::Int, p::ArmonParameters{<:Any, <:CUDADevice})
    print_parameter(io, pad, "GPU", true, nl=false)
    println(io, ": CUDA (block size: $(p.block_size))")
end

end
