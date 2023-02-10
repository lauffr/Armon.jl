
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
        7/5,   # gamma
        1.,    # high_ρ
        0.125, # low_ρ
        1.0,   # high_p
        0.1,   # low_p
        0.,    # high_u
        0.,    # low_u
        0.,    # high_v
        0.,    # low_v
    )
end

function init_test_params(::Bizarrium)
    return (
        2,                 # gamma
        1.42857142857e+4,  # high_ρ
        10000.,            # low_ρ
        6.40939744478e+10, # high_p
        312.5e6,           # low_p
        0.,                # high_u
        250.,              # low_u
        0.,                # high_v
        0.,                # low_v
    )
end

function init_test_params(p::Sedov{T}) where T
    return (
        7/5,   # gamma
        1.,    # high_ρ
        1.,    # low_ρ
        (1.4 - 1) * 0.851072 / (π * p.r^2), # high_p
        1e-14, # low_p
        0.,    # high_u
        0.,    # low_u
        0.,    # high_v
        0.,    # low_v
    )
end

function boundaryCondition(side::Symbol, ::Union{Sod, Bizarrium})::NTuple{2, Int}
    return (side == :left || side == :right) ? (-1, 1) : (1, 1)
end

function boundaryCondition(side::Symbol, ::Sod_y)::NTuple{2, Int}
    return (side == :left || side == :right) ? (1, 1) : (1, -1)
end

function boundaryCondition(side::Symbol, ::Sod_circ)::NTuple{2, Int}
    return (side == :left || side == :right) ? (-1, 1) : (1, -1)
end

function boundaryCondition(::Symbol, ::Sedov)::NTuple{2, Int}
    return (1, 1)
end
