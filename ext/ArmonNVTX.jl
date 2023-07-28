module ArmonNVTX

using Armon
isdefined(Base, :get_extension) ? (import NVTX) : (import ..NVTX)


const armon_domain = NVTX.Domain("Armon")
const range_names = Dict{Symbol, NVTX.StringHandle}()


function NVTX_range_start(::ArmonParameters, name::Symbol)
    # More or less equivalent to NVTX.@range
    # Note that we do not check for NVTX.isactive(). For some reason Nsight Compute might not enable
    # NVTX even when passing '--nvtx', while handling NVTX calls correctly.
    message = get!(range_names, name) do 
        str_hdl = NVTX.StringHandle(armon_domain, string(name))
        NVTX.init!(str_hdl)  # Implicitly initializes 'armon_domain' if needed
        str_hdl
    end
    color = hash(name) % UInt32
    return NVTX.range_start(armon_domain; message, color)
end


function NVTX_range_end(::ArmonParameters, ::Symbol, state)
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
