
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
