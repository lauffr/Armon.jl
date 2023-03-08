
abstract type TestCase end
abstract type TwoStateTestCase <: TestCase end

struct Sod       <: TwoStateTestCase end
struct Sod_y     <: TwoStateTestCase end
struct Sod_circ  <: TwoStateTestCase end
struct Bizarrium <: TwoStateTestCase end
struct Sedov{T}  <: TwoStateTestCase
    r::T
end

create_test(::T, ::T, ::Type{Test}) where {T, Test <: TestCase} = Test()

function create_test(Δx::T, Δy::T, ::Type{Sedov}) where T
    r_Sedov::T = sqrt(Δx^2 + Δy^2) / sqrt(2)
    return Sedov{T}(r_Sedov)
end

test_from_name(::Val{:Sod})       = Sod
test_from_name(::Val{:Sod_y})     = Sod_y
test_from_name(::Val{:Sod_circ})  = Sod_circ
test_from_name(::Val{:Bizarrium}) = Bizarrium
test_from_name(::Val{:Sedov})     = Sedov

test_from_name(::Val{s}) where s = error("Unknown test case: '$s'")
test_from_name(s::Symbol) = test_from_name(Val(s))

test_name(::Test) where {Test <: TestCase} = Test.name.name

default_domain_size(::Type{<:TestCase}) = (1, 1)
default_domain_size(::Type{Sedov}) = (2, 2)

default_domain_origin(::Type{<:TestCase}) = (0, 0)
default_domain_origin(::Type{Sedov}) = (-1, -1)

default_CFL(::Union{Sod, Sod_y, Sod_circ}) = 0.95
default_CFL(::Bizarrium) = 0.6
default_CFL(::Sedov) = 0.7

default_max_time(::Union{Sod, Sod_y, Sod_circ}) = 0.20
default_max_time(::Bizarrium) = 80e-6
default_max_time(::Sedov) = 1.0

Base.show(io::IO, ::Sod)       = print(io, "Sod shock tube")
Base.show(io::IO, ::Sod_y)     = print(io, "Sod shock tube (along the Y axis)")
Base.show(io::IO, ::Sod_circ)  = print(io, "Sod shock tube (cylindrical symmetry around the Z axis)")
Base.show(io::IO, ::Bizarrium) = print(io, "Bizarrium")
Base.show(io::IO, ::Sedov)     = print(io, "Sedov")

# TODO : use 0.0625 for Sod_circ since 1/8 makes no sense and is quite arbitrary
test_region_high(x::T, _::T, ::Sod)       where T = x ≤ 0.5
test_region_high(_::T, y::T, ::Sod_y)     where T = y ≤ 0.5
test_region_high(x::T, y::T, ::Sod_circ)  where T = (x - T(0.5))^2 + (y - T(0.5))^2 ≤ T(0.125)
test_region_high(x::T, _::T, ::Bizarrium) where T = x ≤ 0.5
test_region_high(x::T, y::T, s::Sedov{T}) where T = x^2 + y^2 ≤ s.r^2

function init_test_params(::Union{Sod, Sod_y, Sod_circ})
    return (
        #= γ      =# 7/5,
        #= high_ρ =# 1.,
        #= low_ρ  =# 0.125,
        #= high_E =# 2.5,
        #= low_E  =# 2.0,
        #= high_u =# 0.,
        #= low_u  =# 0.,
        #= high_v =# 0.,
        #= low_v  =# 0.
    )
end

function init_test_params(::Bizarrium)
    return (
        #= γ      =# 2,
        #= high_ρ =# 1.42857142857e+4,
        #= low_ρ  =# 10000.,
        #= high_E =# 4.48657821135e+6,
        #= low_E  =# 0.5 * 250^2,
        #= high_u =# 0.,
        #= low_u  =# 250.,
        #= high_v =# 0.,
        #= low_v  =# 0.
    )
end

function init_test_params(p::Sedov)
    return (
        #= γ      =# 7/5,
        #= high_ρ =# 1.,
        #= low_ρ  =# 1.,
        #= high_E =# 0.851072 / (π * p.r^2),
        #= low_E  =# 2.5e-14,
        #= high_u =# 0.,
        #= low_u  =# 0.,
        #= high_v =# 0.,
        #= low_v  =# 0.
    )
end

function boundaryCondition(side::Side, ::Sod)::NTuple{2, Int}
    return (side == Left || side == Right) ? (-1, 1) : (1, 1)
end

function boundaryCondition(side::Side, ::Sod_y)::NTuple{2, Int}
    return (side == Left || side == Right) ? (1, 1) : (1, -1)
end

function boundaryCondition(side::Side, ::Sod_circ)::NTuple{2, Int}
    return (side == Left || side == Right) ? (-1, 1) : (1, -1)
end

function boundaryCondition(side::Side, ::Bizarrium)::NTuple{2, Int}
    if side == Left
        return (-1, 1)
    elseif side == Right
        return (1, 1)
    else
        return (1, -1)
    end
end

function boundaryCondition(::Side, ::Sedov)::NTuple{2, Int}
    return (1, 1)
end
