
#
# Kernel profiling
#

const KernelCallback = NamedTuple{(:name, :start, :end), Tuple{Symbol, <:Function, <:Function}}
const KERNEL_PROFILING_CALLBACKS = Vector{KernelCallback}()


function register_kernel_callback(cb::KernelCallback; first=false)
    if first
        pushfirst!(KERNEL_PROFILING_CALLBACKS, cb)
    else
        push!(KERNEL_PROFILING_CALLBACKS, cb)
    end
end


function kernel_start(params::ArmonParameters, name::Symbol)
    cb_states = Vector{Any}(undef, length(KERNEL_PROFILING_CALLBACKS))  # TODO: replace by a SVector ?
    for (i, cb) in enumerate(KERNEL_PROFILING_CALLBACKS)
        !(cb.name in params.profiling_info) && continue
        cb_states[i] = cb.start(params, name)
    end
    return cb_states
end


function kernel_end(params::ArmonParameters, name::Symbol, cb_states::Vector{Any})
    for (i, cb) in enumerate(KERNEL_PROFILING_CALLBACKS) |> Iterators.reverse
        !isassigned(cb_states, i) && continue
        cb.end(params, name, cb_states[i])
    end
end

#
# Section profiling
#

const SectionCallback = NamedTuple{(:name, :start, :end), Tuple{Symbol, <:Function, <:Function}}
const SECTION_PROFILING_CALLBACKS = Vector{SectionCallback}()


function register_section_callback(cb::SectionCallback; first=false)
    if first
        pushfirst!(SECTION_PROFILING_CALLBACKS, cb)
    else
        push!(SECTION_PROFILING_CALLBACKS, cb)
    end
end


function section_start(params::ArmonParameters, name::Symbol)
    cb_states = Vector{Any}(undef, length(SECTION_PROFILING_CALLBACKS))  # TODO: replace by a SVector ?
    for (i, cb) in enumerate(SECTION_PROFILING_CALLBACKS)
        !(cb.name in params.profiling_info) && continue
        cb_states[i] = cb.start(params, name)
    end
    return cb_states
end


function section_end(params::ArmonParameters, name::Symbol, cb_states::Vector{Any})
    for (i, cb) in enumerate(SECTION_PROFILING_CALLBACKS) |> Iterators.reverse
        !isassigned(cb_states, i) && continue
        cb.end(params, name, cb_states[i])
    end
end


function section(name, expr; force_async=false)
    @gensym section_state profiling_on section_name result
    return esc(quote
        $profiling_on = params.enable_profiling
        if $profiling_on
            $section_name = Symbol($name)
            $section_state = section_start(params, $section_name)
        end

        local $result
        # tryfinally has the advantage to not introduce a new scope, unlike an explicit try-finally
        # block.
        $(Expr(:tryfinally, quote
            # try
            $result = $expr
            if $(!force_async) && !params.time_async
                wait(params)
            end
        end, quote
            # finally
            if $profiling_on
                section_end(params, $section_name, $section_state)
            end
        end))

        $result
    end)
end


macro section(name, expr)
    return section(name, expr)
end


macro section(name, option, expr)
    if !@capture(option, kw_ = val_)
        error("Expected an expression of the form 'kw=val', got: $option")
    end
    (kw !== :async) && error("Unknown option: $kw")
    val = Core.eval(__module__, val)
    return section(name, expr;
        force_async=val
    )
end

#
# TimerOutputs profiling
#

function timeit_section_start(params::ArmonParameters, name::Symbol)
    # More or less equivalent to @timeit params.timer name $expr
    if params.timer.enabled
        data = TimerOutputs.push!(params.timer, string(name))
    else
        # For type stability
        data = params.timer.accumulated_data
    end
    b₀ = TimerOutputs.gc_bytes()
    t₀ = TimerOutputs.time_ns()
    return (data, b₀, t₀)
end


function timeit_section_end(params::ArmonParameters, ::Symbol, state)
    if params.timer.enabled
        (data, b₀, t₀) = state
        TimerOutputs.do_accumulate!(data, t₀, b₀)
        TimerOutputs.pop!(params.timer)
    end
end


register_section_callback(SectionCallback((
    :TimerOutputs,
    timeit_section_start,
    timeit_section_end
)))
