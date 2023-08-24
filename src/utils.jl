
"""
    Axis

Enumeration of the axes of the domain
"""
@enum Axis X_axis Y_axis


"""
    Side

Enumeration of the sides of the domain
"""
@enum Side Left Right Bottom Top

sides_along(dir::Axis) = dir == X_axis ? (Left, Right) : (Bottom, Top)

function offset_to(side::Side)
    if side == Left
        return (-1, 0)
    elseif side == Right
        return (1, 0)
    elseif side == Bottom
        return (0, -1)
    elseif side == Top
        return (0, 1)
    end
end


"""
    CPU_HP

Device tag for the high-performance CPU backend using multithreading (with Polyester.jl) and
vectorisation.
"""
struct CPU_HP end


"""
    SolverException

Thrown when the solver encounters an invalid state.

The `category` field can be used to distinguish between error types without inspecting the error
message:
 - `:config`: a problem in the solver configuration, usually thrown when constructing `ArmonParameters`
 - `:cpp`: a C++ exception thrown by the C++ Kokkos backend
 - `:time`: an invalid time step

`ErrorException`s thrown by the solver represent internal errors.
"""
struct SolverException <: Exception
    category::Symbol
    msg::String
end


solver_error(category::Symbol, msg::String) = throw(SolverException(category, msg))

function solver_error(category::Symbol, msgs::Vararg{Any, N}) where {N}
    throw(SolverException(category, Base.string(msgs...)))
end


function Base.showerror(io::IO, ex::SolverException)
    print(io, "SolverException ($(ex.category)): ", ex.msg)
end


"""
    @reuse_tls(storage, expr)

Reuses the TLS of the task returned by `expr`. The TLS is stored in `storage`.
If `storage` is `nothing`, it is set to the tasks' TLS, otherwise, the task's TLS is set to
`storage`.

`storage` must evaluate to a value of type `Nothing` or `IdDict{Any, Any}`.

`expr` can be a `Threads.@spawn` or an `@async` expression, or a `Task`.
After setting the tasks' storage, it is scheduled.

Compatible with `@sync`.

```julia
julia> my_tls = nothing

julia> @reuse_tls my_tls Threads.@spawn begin task_local_storage(:a, 1) end;

julia> my_tls
IdDict{Any, Any} with 1 entry:
  :a => 1

```
"""
macro reuse_tls(storage, expr)
    uses_schedule = false
    MacroTools.postwalk(expr) do e
        (e isa Expr && e.head === :macrocall) || return e
        uses_schedule |= first(e.args) in (:(Threads.var"@spawn"), Symbol("@spawn"), Symbol("@async"))
        return e
    end

    if uses_schedule
        expr = macroexpand(__module__, expr, recursive=false)

        # Remove the `schedule` call from the macro. It is wrapped in a `GlobalRef`, making it 
        # cumbersome to find.

        # `Base.schedule` in `@async`
        schedule_binding_1 = GlobalRef(Base, :schedule)
        schedule_call_1 = :($schedule_binding_1(t_))

        # `Base.Threads.schedule` in `Threads.@spawn`
        schedule_binding_2 = GlobalRef(Base.Threads, :schedule)
        schedule_call_2 = :($schedule_binding_2(t_))

        schedule_removed = false
        expr = MacroTools.postwalk(expr) do e
            if isnothing(MacroTools.trymatch(schedule_call_1, e)) &&
                    isnothing(MacroTools.trymatch(schedule_call_2, e))
                return e
            else
                schedule_removed = true
                return nothing
            end
        end
        @assert schedule_removed
    else
        expr = :($expr::Task)
    end

    task_var = gensym(:task)
    return esc(quote
        local $task_var = $expr
        if Base.isnothing($storage)
            $storage = Base.get_task_tls($task_var)
        else
            $task_var.storage = $storage::Base.IdDict{Any, Any}
        end
        Base.Threads.schedule($task_var)
        $task_var
    end)
end
