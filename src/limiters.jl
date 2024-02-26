
struct NoLimiter       <: Limiter end
struct MinmodLimiter   <: Limiter end
struct SuperbeeLimiter <: Limiter end

@kernel_function limiter(_::T, ::NoLimiter)       where T = one(T)
@kernel_function limiter(r::T, ::MinmodLimiter)   where T = max(zero(T), min(one(T), r))
@kernel_function limiter(r::T, ::SuperbeeLimiter) where T = max(zero(T), min(2r, one(T)), min(r, 2*one(T)))

limiter_from_name(::Val{:no_limiter}) = NoLimiter()
limiter_from_name(::Val{:minmod})     = MinmodLimiter()
limiter_from_name(::Val{:superbee})   = SuperbeeLimiter()

limiter_from_name(::Val{s}) where s = solver_error(:config, "Unknown limiter name: '$s'")
limiter_from_name(s::Symbol) = limiter_from_name(Val(s))

Base.show(io::IO, ::NoLimiter)       = print(io, "No limiter")
Base.show(io::IO, ::MinmodLimiter)   = print(io, "Minmod limiter")
Base.show(io::IO, ::SuperbeeLimiter) = print(io, "Superbee limiter")
