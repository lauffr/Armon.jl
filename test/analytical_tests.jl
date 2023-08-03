
using Printf
using Armon
import Armon: Axis, InitTestParams


abstract type AnalyticTestCase <: Armon.TestCase end

struct AnalyticalGravity1D{T} <: AnalyticTestCase
    g::T
    x₀::T
end

struct AnalyticalGravity2D{T} <: AnalyticTestCase
    g::T
    x₀::T
    y₀::T
end

struct AnalyticalFriction{T} <: AnalyticTestCase
    λ::T
    α::T
end

function Armon.create_test(::T, ::T, ::Type{AnalyticalGravity1D}) where T
    g::T = 9.82
    x₀::T = 1.5
    return AnalyticalGravity1D{T}(g, x₀)
end

function Armon.create_test(::T, ::T, ::Type{AnalyticalGravity2D}) where T
    g::T = 9.82
    x₀::T = 1.5
    y₀::T = 1.5
    return AnalyticalGravity2D{T}(g, x₀, y₀)
end

function Armon.create_test(_::T, _::T, ::Type{AnalyticalFriction}) where T
    λ::T = 1
    γ::T = 7/5
    α::T = γ / (γ - 1)
    return AnalyticalFriction{T}(λ, α)
end


Armon.test_from_name(::Val{:Gravity1D}) = AnalyticalGravity1D
Armon.test_from_name(::Val{:Gravity2D}) = AnalyticalGravity2D
Armon.test_from_name(::Val{:Friction})  = AnalyticalFriction

Armon.default_CFL(::AnalyticTestCase) = 0.25
Armon.default_max_time(::AnalyticTestCase) = 0.1
Armon.is_conservative(::AnalyticTestCase) = false
Armon.has_source_term(::AnalyticTestCase) = true

Base.show(io::IO, ::AnalyticalGravity1D) = print(io, "Gravity 1D")
Base.show(io::IO, ::AnalyticalGravity2D) = print(io, "Gravity 2D")
Base.show(io::IO, ::AnalyticalFriction)  = print(io, "Friction")


function Armon.boundaryCondition(side::Armon.Side, ::AnalyticTestCase)::NTuple{2, Int}
    return (side == Armon.Left || side == Armon.Right) ? Armon.BC_Dirichlet_X : Armon.BC_Dirichlet_Y
end


function Armon.update_test_params(t::AnalyticalGravity1D{T}, axis::Axis) where {T}
    return if axis == X_axis
        AnalyticalGravity1D{T}(t.g)
    else
        AnalyticalGravity1D{T}(zero(T))
    end
end


function Armon.update_test_params(t::AnalyticalGravity2D{T}, _::Axis) where {T}
    return AnalyticalGravity2D{T}(t.g, t.y₀, t.x₀)
end


Armon.@kernel_function function Armon.source_term(::T, ρ::T, u::T, test::AnalyticalGravity1D{T}) where {T}
    return (
        ρ = zero(T),
        u = -test.g,
        E = ρ * -test.g * u
    )
end

Armon.@kernel_function function Armon.source_term(x::T, ρ::T, u::T, test::AnalyticalGravity2D{T}) where {T}
    ∇ζ = -test.g * 2 * (x - test.x₀)
    return (
        ρ = zero(T),
        u = ∇ζ,
        E = ρ * ∇ζ * u
    )
end

Armon.@kernel_function function Armon.source_term(::T, ρ::T, u::T, test::AnalyticalFriction{T}) where {T}
    return (
        ρ = zero(T),
        u = -test.λ * ρ * u,
        E = -test.α * test.λ * ρ * u^2
    )
end


Armon.@kernel_function function Armon.init_test_params(x::T, ::T, test::AnalyticalGravity1D{T}) where {T}
    γ::T = 7/5
    return InitTestParams{T}(
        #= ρ =# 2.,
        #= E =# 2 * test.g * (test.x₀ - x) / (γ - 1) / 2.,  # P = (γ - 1) ρ E
        #= u =# 0.,
        #= v =# 0.
    )
end

Armon.@kernel_function function Armon.init_test_params(x::T, y::T, test::AnalyticalGravity2D{T}) where {T}
    γ::T = 7/5
    ζ::T = -test.g * ((x - test.x₀)^2 + (y - test.y₀)^2)
    return InitTestParams{T}(
        #= ρ =# exp(ζ),
        #= E =# 1 / (γ - 1),
        #= u =# 0.,
        #= v =# 0.
    )
end

Armon.@kernel_function function Armon.init_test_params(x::T, y::T, test::AnalyticalFriction{T}) where {T}
    γ::T = 7/5
    λ = test.λ
    return InitTestParams{T}(
        #= ρ =# 1,
        #= E =# -λ / (γ - 1) * (x + y) / √2 + 2λ / (γ - 1) + 1,
        #= u =# 1 / √2,
        #= v =# 1 / √2
    )
end


Base.length(::AnalyticTestCase) = 1
Base.iterate(t::AnalyticTestCase) = t

reduce_error(res, err) = (first(res) + err^2, max(last(res), abs(err)))

function comp_error(params::ArmonParameters, data::Armon.ArmonData, var::Symbol)
    (; x, y, domain_mask) = data

    r = params.ideb:params.ifin
    test = params.test

    if var === :ρ
        result = data.rho
    elseif var === :u
        result = data.umat
    elseif var === :v
        result = data.vmat
    elseif var === :E
        result = data.Emat
    else
        error("oops: $var")
    end

    err_L₂, err_L∞ = 0., 0.
    for i in r
        var_exact = getfield(Armon.init_test_params(x[i], y[i], test), var)
        err = (result[i] - var_exact) * domain_mask[i]
        err_L₂, err_L∞ = reduce_error((err_L₂, err_L∞), err)
    end
    err_L₂ = sqrt(err_L₂)

    return err_L₂, err_L∞
end


function analytical_test(params::ArmonParameters; kwargs...)
    params.return_data = true
    state = armon(params)

    for var in (:ρ, :u, :v, :E)
        err_L₂, err_L∞ = comp_error(params, Armon.device(state.data), var)
        @printf("Error for %s: %15.8g (L₂), %15.8g (L∞)\n", var, err_L₂, err_L∞)
    end
end
