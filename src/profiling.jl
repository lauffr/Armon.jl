
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
    return quote
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
    end
end


function section_for_loop(name, expr; force_async=false)
    loop_body = expr.args[2]
    expr.args[2] = section(name, loop_body; force_async)
    return expr
end


"""
    @section(name, expr)
    @section(name, options, expr)

Introduce a profiling section around `expr`. Sections can be nested. Sections do not introduce a new
scope.

Placed before a for-loop, a new section will be started for each iteration. `name` can interpolate
using loop variables (like `Test.@testset`).

It is assumed that a `params` variable of `ArmonParameters` is present in the scope of the `@section`.

`options` is of the form `key=value`:
 - `async` (default: `false`): if `async=false` (and `!params.time_async`), a barrier (`wait(params)`)
   is added at the end of the section.

```julia
params = ArmonParameters(#= ... =#)

@section "Iteration \$i" for i in 1:10
    j = @section "Foo" begin
        foo(i)
    end

    @section "Some calculation" begin
        k = bar(i, j)
    end

    @sync begin
        @async begin
            @section "Task 1" async=true my_task_1(i, j, k)
        end

        @async begin
            @section "Task 2" async=true my_task_2(i, j, k)
        end
    end
end
```
"""
macro section(args...)
    2 ≤ length(args) ≤ 3 || error("invalid number of arguments to @section")

    name = args[1]
    expr = args[end]
    option = length(args) == 3 ? args[2] : nothing

    if !isnothing(option)
        if !@capture(option, kw_ = val_)
            error("Expected an expression of the form 'kw=val', got: $option")
        end
        (kw !== :async) && error("Unknown option: $kw")
        val = Core.eval(__module__, val)
        force_async = val
    else
        force_async = false
    end

    if !isa(expr, Expr)
        error("expected expression or for-loop, got: $(type(expr))")
    elseif expr.head == :for
        expr = section_for_loop(name, expr; force_async)
    else
        expr = section(name, expr; force_async)
    end

    return esc(expr)
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
