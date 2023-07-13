module ArmonNVTX

using Armon
isdefined(Base, :get_extension) ? (import NVTX) : (import ..NVTX)


const armon_domain = NVTX.Domain("Armon")


function NVTX_range_start(::ArmonParameters, name::Symbol)
    !NVTX.isactive() && return nothing

    # More or less equivalent to NVTX.@range
    NVTX.init!(Armon_domain)
    message = NVTX.StringHandle(Armon_domain, string(name))
    color = hash(name) % UInt32

    return NVTX.range_start(Armon_domain; message, color)
end


function NVTX_range_end(::ArmonParameters, ::Symbol, state)
    isnothing(state) && return
    NVTX.range_end(state)
end


function __init__()
    Armon.register_kernel_callback(Armon.KernelCallback((
        :NVTX_kernels,
        NVTX_range_start,
        NVTX_range_end,
    )); first=true)
    
    Armon.register_section_callback(Armon.SectionCallback((
        :NVTX_sections,
        NVTX_range_start,
        NVTX_range_end
    )); first=true)
end

end
