
abstract type Limiter end

struct NoLimiter       <: Limiter end
struct MinmodLimiter   <: Limiter end
struct SuperbeeLimiter <: Limiter end

# TODO: apply @kernel_function to the limiters (requires most likely to separate 'generic_kernel.jl' into its own package)
limiter(_::T, ::NoLimiter)       where T = one(T)
limiter(r::T, ::MinmodLimiter)   where T = max(zero(T), min(one(T), r))
limiter(r::T, ::SuperbeeLimiter) where T = max(zero(T), min(2r, one(T)), min(r, 2*one(T)))

limiter_from_name(::Val{:no_limiter}) = NoLimiter()
limiter_from_name(::Val{:minmod})     = MinmodLimiter()
limiter_from_name(::Val{:superbee})   = SuperbeeLimiter()

limiter_from_name(::Val{s}) where s = error("Unknown limiter name: '$s'")
limiter_from_name(s::Symbol) = limiter_from_name(Val(s))

Base.show(io::IO, ::NoLimiter)       = print(io, "No limiter")
Base.show(io::IO, ::MinmodLimiter)   = print(io, "Minmod limiter")
Base.show(io::IO, ::SuperbeeLimiter) = print(io, "Superbee limiter")
