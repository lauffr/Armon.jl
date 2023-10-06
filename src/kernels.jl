
@kernel_function function acoustic_Godunov(ρᵢ::T, ρᵢ₋₁::T, cᵢ::T, cᵢ₋₁::T, uᵢ::T, uᵢ₋₁::T, pᵢ::T, pᵢ₋₁::T) where T
    rc_l = ρᵢ₋₁ * cᵢ₋₁
    rc_r = ρᵢ   * cᵢ
    ustarᵢ = (rc_l * uᵢ₋₁ + rc_r * uᵢ +               (pᵢ₋₁ - pᵢ)) / (rc_l + rc_r)
    pstarᵢ = (rc_r * pᵢ₋₁ + rc_l * pᵢ + rc_l * rc_r * (uᵢ₋₁ - uᵢ)) / (rc_l + rc_r)
    return ustarᵢ, pstarᵢ
end


@generic_kernel function acoustic!(
    s::Int, ustar_::V, pstar_::V,
    rho::V, u::V, pmat::V, cmat::V
) where V
    i = @index_2D_lin()
    ustar_[i], pstar_[i] = acoustic_Godunov(
        rho[i], rho[i-s], cmat[i], cmat[i-s],
          u[i],   u[i-s], pmat[i], pmat[i-s]
    )
end


@generic_kernel function acoustic_GAD!(
    s::Int, dt::T, dx::T,
    ustar::V, pstar::V, rho::V, u::V, pmat::V, cmat::V,
    ::LimiterType
) where {T, V <: AbstractArray{T}, LimiterType <: Limiter}
    i = @index_2D_lin()

    # First order acoustic solver on the left cell
    ustar_i₋, pstar_i₋ = acoustic_Godunov(
        rho[i-s], rho[i-2s], cmat[i-s], cmat[i-2s],
          u[i-s],   u[i-2s], pmat[i-s], pmat[i-2s]
    )

    # First order acoustic solver on the current cell
    ustar_i, pstar_i = acoustic_Godunov(
        rho[i], rho[i-s], cmat[i], cmat[i-s],
          u[i],   u[i-s], pmat[i], pmat[i-s]
    )

    # First order acoustic solver on the right cell
    ustar_i₊, pstar_i₊ = acoustic_Godunov(
        rho[i+s], rho[i], cmat[i+s], cmat[i],
          u[i+s],   u[i], pmat[i+s], pmat[i]
    )

    # Second order GAD acoustic solver on the current cell

    r_u₋ = (ustar_i₊ -      u[i]) / (ustar_i -    u[i-s] + T(1e-6))
    r_p₋ = (pstar_i₊ -   pmat[i]) / (pstar_i - pmat[i-s] + T(1e-6))
    r_u₊ = (   u[i-s] - ustar_i₋) / (   u[i] -   ustar_i + T(1e-6))
    r_p₊ = (pmat[i-s] - pstar_i₋) / (pmat[i] -   pstar_i + T(1e-6))

    r_u₋ = limiter(r_u₋, LimiterType())
    r_p₋ = limiter(r_p₋, LimiterType())
    r_u₊ = limiter(r_u₊, LimiterType())
    r_p₊ = limiter(r_p₊, LimiterType())

    dm_l = rho[i-s] * dx
    dm_r = rho[i]   * dx
    Dm   = (dm_l + dm_r) / 2

    rc_l = rho[i-s] * cmat[i-s]
    rc_r = rho[i]   * cmat[i]
    θ    = T(0.5) * (1 - (rc_l + rc_r) / 2 * (dt / Dm))

    ustar[i] = ustar_i + θ * (r_u₊ * (   u[i] - ustar_i) - r_u₋ * (ustar_i -    u[i-s]))
    pstar[i] = pstar_i + θ * (r_p₊ * (pmat[i] - pstar_i) - r_p₋ * (pstar_i - pmat[i-s]))
end


@generic_kernel function perfect_gas_EOS!(
    gamma::T, 
    rho::V, Emat::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V
) where {T, V <: AbstractArray{T}}
    i = @index_2D_lin()
    e = Emat[i] - 0.5 * (umat[i]^2 + vmat[i]^2)
    pmat[i] = (gamma - 1.) * rho[i] * e
    cmat[i] = sqrt(gamma * pmat[i] / rho[i])
    gmat[i] = (1. + gamma) / 2
end


@generic_kernel function bizarrium_EOS!(
    rho::V, umat::V, vmat::V, Emat::V, pmat::V, cmat::V, gmat::V
) where {T, V <: AbstractArray{T}}
    i = @index_2D_lin()

    # O. Heuzé, S. Jaouen, H. Jourdren, 
    # "Dissipative issue of high-order shock capturing schemes with non-convex equations of state"
    # JCP 2009

    @kernel_init begin
        rho0::T = 10000.
        K0::T   = 1e+11
        Cv0::T  = 1000.
        T0::T   = 300.
        eps0::T = 0.
        G0::T   = 1.5
        s::T    = 1.5
        q::T    = -42080895/14941154
        r::T    = 727668333/149411540
    end

    x = rho[i] / rho0 - 1
    g = G0 * (1-rho0 / rho[i])

    f0 = (1+(s/3-2)*x+q*x^2+r*x^3)/(1-s*x)
    f1 = (s/3-2+2*q*x+3*r*x^2+s*f0)/(1-s*x)
    f2 = (2*q+6*r*x+2*s*f1)/(1-s*x)
    f3 = (6*r+3*s*f2)/(1-s*x)

    epsk0     = eps0 - Cv0*T0*(1+g) + 0.5*(K0/rho0)*x^2*f0
    pk0       = -Cv0*T0*G0*rho0 + 0.5*K0*x*(1+x)^2*(2*f0+x*f1)
    pk0prime  = -0.5*K0*(1+x)^3*rho0 * (2*(1+3x)*f0 + 2*x*(2+3x)*f1 + x^2*(1+x)*f2)
    pk0second = 0.5*K0*(1+x)^4*rho0^2 * (12*(1+2x)*f0 + 6*(1+6x+6*x^2)*f1 + 
                                                    6*x*(1+x)*(1+2x)*f2 + x^2*(1+x)^2*f3)

    e = Emat[i] - 0.5 * (umat[i]^2 + vmat[i]^2)
    pmat[i] = pk0 + G0 * rho0 * (e - epsk0)
    cmat[i] = sqrt(G0 * rho0 * (pmat[i] - pk0) - pk0prime) / rho[i]
    gmat[i] = 0.5 / (rho[i]^3 * cmat[i]^2) * (pk0second + (G0 * rho0)^2 * (pmat[i] - pk0))
end


@generic_kernel function cell_update!(
    s::Int, dx::T, dt::T, 
    ustar::V, pstar::V, rho::V, u::V, Emat::V
) where {T, V <: AbstractArray{T}}
    i = @index_2D_lin()
    dm = rho[i] * dx
    rho[i]   = dm / (dx + dt * (ustar[i+s] - ustar[i]))
    u[i]    += dt / dm * (pstar[i]            - pstar[i+s]             )
    Emat[i] += dt / dm * (pstar[i] * ustar[i] - pstar[i+s] * ustar[i+s])
end


@generic_kernel function cell_update_with_source!(
    s::Int, dx::T, dt::T, 
    ustar::V, pstar::V, x_::V, rho::V, u::V, Emat::V, test_case::Test
) where {T, V <: AbstractArray{T}, Test <: TestCase}
    i = @index_2D_lin()
    Δx = dx + dt * (ustar[i+s] - ustar[i])  # Length of the lagrangian cell
    source = source_term(x_[i] + Δx / 2, rho[i], u[i], test_case)::SourceTermType{T}
    dm = rho[i] * dx
    rho[i]   = dm / Δx                                                   + source.ρ * dt
    u[i]    += dt / dm * (pstar[i]            - pstar[i+s]             ) + source.u * dt
    Emat[i] += dt / dm * (pstar[i] * ustar[i] - pstar[i+s] * ustar[i+s]) + source.E * dt
end


@generic_kernel function euler_projection!(
    s::Int, dx::T, dt::T,
    ustar::V, rho::V, umat::V, vmat::V, Emat::V,
    advection_ρ::V, advection_uρ::V, advection_vρ::V, advection_Eρ::V
) where {T, V <: AbstractArray{T}}
    i = @index_2D_lin()

    dX = dx + dt * (ustar[i+s] - ustar[i])

    tmp_ρ  = (dX * rho[i]           - (advection_ρ[i+s]  - advection_ρ[i] )) / dx
    tmp_uρ = (dX * rho[i] * umat[i] - (advection_uρ[i+s] - advection_uρ[i])) / dx
    tmp_vρ = (dX * rho[i] * vmat[i] - (advection_vρ[i+s] - advection_vρ[i])) / dx
    tmp_Eρ = (dX * rho[i] * Emat[i] - (advection_Eρ[i+s] - advection_Eρ[i])) / dx

    rho[i]  = tmp_ρ
    umat[i] = tmp_uρ / tmp_ρ
    vmat[i] = tmp_vρ / tmp_ρ
    Emat[i] = tmp_Eρ / tmp_ρ
end


@generic_kernel function advection_first_order!(
    s::Int, dt::T,
    ustar::V, rho::V, umat::V, vmat::V, Emat::V,
    advection_ρ::V, advection_uρ::V, advection_vρ::V, advection_Eρ::V
) where {T, V <: AbstractArray{T}}
    i = @index_2D_lin()
    is = i
    disp = dt * ustar[i]
    if disp > 0
        i = i - s
    end

    advection_ρ[is]  = disp * (rho[i]          )
    advection_uρ[is] = disp * (rho[i] * umat[i])
    advection_vρ[is] = disp * (rho[i] * vmat[i])
    advection_Eρ[is] = disp * (rho[i] * Emat[i])
end


@generic_kernel function advection_second_order!(
    s::Int, dx::T, dt::T,
    ustar::V, rho::V, umat::V, vmat::V, Emat::V,
    advection_ρ::V, advection_uρ::V, advection_vρ::V, advection_Eρ::V
) where {T, V <: AbstractArray{T}}
    i = @index_2D_lin()
    is = i
    disp = dt * ustar[i]
    if disp > 0
        Δxₑ = -(dx - dt * ustar[i-s])
        i = i - s
    else
        Δxₑ = dx + dt * ustar[i+s]
    end

    Δxₗ₋  = dx + dt * (ustar[i]    - ustar[i-s])
    Δxₗ   = dx + dt * (ustar[i+s]  - ustar[i]  )
    Δxₗ₊  = dx + dt * (ustar[i+2s] - ustar[i+s])

    r₋  = (2 * Δxₗ) / (Δxₗ + Δxₗ₋)
    r₊  = (2 * Δxₗ) / (Δxₗ + Δxₗ₊)

    slopes_ρ  = slope_minmod(rho[i-s]            , rho[i]          , rho[i+s]            , r₋, r₊)
    slopes_uρ = slope_minmod(rho[i-s] * umat[i-s], rho[i] * umat[i], rho[i+s] * umat[i+s], r₋, r₊)
    slopes_vρ = slope_minmod(rho[i-s] * vmat[i-s], rho[i] * vmat[i], rho[i+s] * vmat[i+s], r₋, r₊)
    slopes_Eρ = slope_minmod(rho[i-s] * Emat[i-s], rho[i] * Emat[i], rho[i+s] * Emat[i+s], r₋, r₊)

    length_factor = Δxₑ / (2 * Δxₗ)
    advection_ρ[is]  = disp * (rho[i]           - slopes_ρ  * length_factor)
    advection_uρ[is] = disp * (rho[i] * umat[i] - slopes_uρ * length_factor)
    advection_vρ[is] = disp * (rho[i] * vmat[i] - slopes_vρ * length_factor)
    advection_Eρ[is] = disp * (rho[i] * Emat[i] - slopes_Eρ * length_factor)
end


@generic_kernel function init_test(
    row_length::Int, nghost::Int, nx::Int, ny::Int, 
    domain_size::NTuple{2, T}, origin::NTuple{2, T},
    cart_coords::NTuple{2, Int}, global_grid::NTuple{2, Int},
    x::V, y::V, rho::V, Emat::V, umat::V, vmat::V, 
    domain_mask::V, pmat::V, cmat::V, ustar::V, pstar::V, 
    test_case::Test, debug_indexes::Bool
) where {T, V <: AbstractArray{T}, Test <: TestCase}
    @kernel_init begin
        (cx, cy) = cart_coords
        (g_nx, g_ny) = global_grid
        (sx, sy) = domain_size
        (ox, oy) = origin

        # Position of the origin of this sub-domain
        pos_x = cx * nx
        pos_y = cy * ny

        if Test <: TwoStateTestCase
            (; high_ρ::T, low_ρ::T,
               high_E::T, low_E::T,
               high_u::T, low_u::T,
               high_v::T, low_v::T) = init_test_params(test_case)::InitTestParamsTwoState{T}
        end
    end

    i = @index_1D_lin()

    ix = ((i-1) % row_length) - nghost
    iy = ((i-1) ÷ row_length) - nghost

    # Global indexes, used only to know to compute the position of the cell
    g_ix = ix + pos_x
    g_iy = iy + pos_y

    x[i] = g_ix / g_nx * sx + ox
    y[i] = g_iy / g_ny * sy + oy

    x_mid = x[i] + sx / (2*g_nx)
    y_mid = y[i] + sy / (2*g_ny)

    if debug_indexes
        rho[i]  = i
        Emat[i] = i
        umat[i] = i
        vmat[i] = i
    else
        if Test <: TwoStateTestCase
            if test_region_high(x_mid, y_mid, test_case)
                rho[i]  = high_ρ
                Emat[i] = high_E
                umat[i] = high_u
                vmat[i] = high_v
            else
                rho[i]  = low_ρ
                Emat[i] = low_E
                umat[i] = low_u
                vmat[i] = low_v
            end
        else
            init_vals::InitTestParams{T} = init_test_params(x_mid, y_mid, test_case)
            rho[i]  = init_vals.ρ
            Emat[i] = init_vals.E
            umat[i] = init_vals.u
            vmat[i] = init_vals.v
        end
    end

    # Set the domain mask to 1 if the cell is real or 0 otherwise
    domain_mask[i] = (0 ≤ ix < nx && 0 ≤ iy < ny) ? 1 : 0

    # TODO: remove this as it should be unnecessary now
    # Set to zero to make sure no non-initialized values changes the result
    pmat[i] = 0
    cmat[i] = 1  # Set to 1 as a max speed of 0 will create NaNs
    ustar[i] = 0
    pstar[i] = 0
end

#
# Wrappers
#

function numerical_fluxes!(
    params::ArmonParameters, data::ArmonDualData, 
    range::DomainRange, label::Symbol
)
    dt = params.cycle_dt
    d_data = device(data)
    u = params.current_axis == X_axis ? d_data.umat : d_data.vmat
    if params.riemann == :acoustic  # 2-state acoustic solver (Godunov)
        if params.scheme == :Godunov
            return acoustic!(params, d_data, range, d_data.ustar, d_data.pstar, u)
        elseif params.scheme == :GAD
            return acoustic_GAD!(params, d_data, range, dt, u, params.riemann_limiter)
        else
            solver_error(:config, "Unknown acoustic scheme: ", params.scheme)
        end
    else
        solver_error(:config, "Unknown Riemann solver: ", params.riemann)
    end
end


function numerical_fluxes!(params::ArmonParameters, data::ArmonDualData, label::Symbol)
    (; steps_ranges) = params

    if label == :inner
        range = steps_ranges.inner_fluxes
    elseif label == :outer_lb
        range = steps_ranges.outer_lb_fluxes
        label = :outer
    elseif label == :outer_rt
        range = steps_ranges.outer_rt_fluxes
        label = :outer
    elseif label == :full
        range = steps_ranges.fluxes
    elseif label == :test
        range = steps_ranges.real_domain
    else
        solver_error(:config, "Wrong region label: $label")
    end

    return numerical_fluxes!(params, data, range, label)
end


function update_EOS!(params::ArmonParameters, data::ArmonData, ::TestCase, range::DomainRange)
    gamma = data_type(params)(7/5)
    return perfect_gas_EOS!(params, data, range, gamma)
end


function update_EOS!(params::ArmonParameters, data::ArmonData, ::Bizarrium, range::DomainRange)
    return bizarrium_EOS!(params, data, range)
end


function update_EOS!(params::ArmonParameters, data::ArmonDualData, label::Symbol)
    (; steps_ranges) = params

    if label == :inner
        range = steps_ranges.inner_EOS
    elseif label == :outer_lb
        range = steps_ranges.outer_lb_EOS
        label = :outer
    elseif label == :outer_rt
        range = steps_ranges.outer_rt_EOS
        label = :outer
    elseif label == :full
        range = steps_ranges.EOS
    elseif label == :test
        range = steps_ranges.real_domain
    else
        solver_error(:config, "Wrong region label: $label")
    end

    return update_EOS!(params, device(data), params.test, range)
end


function init_test(params::ArmonParameters, data::ArmonDualData)
    return init_test(params, device(data), 1:params.nbcell, params.test)
end


function cell_update!(params::ArmonParameters, data::ArmonDualData)
    range = params.steps_ranges.cell_update
    d_data = device(data)
    u = params.current_axis == X_axis ? d_data.umat : d_data.vmat
    x_ = params.current_axis == X_axis ? d_data.x : d_data.y
    if has_source_term(params.test)
        return cell_update_with_source!(params, d_data, range, params.cycle_dt, x_, u, params.test)
    else
        return cell_update!(params, d_data, range, params.cycle_dt, u)
    end
end


@kernel_function function slope_minmod(uᵢ₋::T, uᵢ::T, uᵢ₊::T, r₋::T, r₊::T) where T
    Δu₊ = r₊ * (uᵢ₊ - uᵢ )
    Δu₋ = r₋ * (uᵢ  - uᵢ₋)
    s = sign(Δu₊)
    return s * max(0, min(s * Δu₊, s * Δu₋))
end


function projection_remap!(params::ArmonParameters, data::ArmonDualData)
    d_data = device(data)
    (; work_array_1, work_array_2, work_array_3, work_array_4) = d_data
    advection_ρ  = work_array_1
    advection_uρ = work_array_2
    advection_vρ = work_array_3
    advection_Eρ = work_array_4

    advection_range = params.steps_ranges.advection
    projection_range = params.steps_ranges.projection

    @section "Advection" if params.projection == :euler
        advection_first_order!(params, d_data, advection_range, params.cycle_dt,
            advection_ρ, advection_uρ, advection_vρ, advection_Eρ)
    elseif params.projection == :euler_2nd
        advection_second_order!(params, d_data, advection_range, params.cycle_dt,
            advection_ρ, advection_uρ, advection_vρ, advection_Eρ)
    else
        solver_error(:config, "Unknown projection scheme: $(params.projection)")
    end

    return @section "Projection" euler_projection!(params, d_data, projection_range, params.cycle_dt,
        advection_ρ, advection_uρ, advection_vρ, advection_Eρ)
end


@inline @fast function dtCFL_kernel_reduction(u::T, v::T, c::T, mask::T, dx::T, dy::T) where T
    # We need the absolute value of the divisor since the result of the max can be negative,
    # because of some IEEE 754 non-compliance since fast math is enabled when compiling this code
    # for GPU, e.g.: `@fastmath max(-0., 0.) == -0.`, while `max(-0., 0.) == 0.`
    # If the mask is 0, then: `dx / -0.0 == -Inf`, which will then make the result incorrect.
    return min(
        dx / abs(max(abs(u + c), abs(u - c)) * mask),
        dy / abs(max(abs(v + c), abs(v - c)) * mask)
    )
end


function dtCFL_kernel(::ArmonParameters{T, CPU_HP}, data::ArmonData, range, dx, dy) where T
    (; cmat, umat, vmat, domain_mask) = data

    @batch threadlocal=typemax(T) for i in range
        min_dt = dtCFL_kernel_reduction(umat[i], vmat[i], cmat[i], domain_mask[i], dx, dy)
        threadlocal = min(threadlocal, min_dt)
    end

    return minimum(threadlocal)
end


function dtCFL_kernel(::ArmonParameters{<:Any, <:GPU}, data::ArmonData, range, dx, dy)
    (; cmat, umat, vmat, domain_mask) = data
    return @inbounds reduce(min, @views(
        dtCFL_kernel_reduction.(umat[range], vmat[range], cmat[range], domain_mask[range], dx, dy)
    ))
end


function local_time_step(
    params::ArmonParameters{T}, data::ArmonData{V}, prev_dt::T;
) where {T, V <: AbstractArray{T}}
    (; cfl, ideb, ifin, global_grid, domain_size) = params
    @indexing_vars(params)

    (g_nx, g_ny) = global_grid
    (sx, sy) = domain_size
    dx::T = sx / g_nx
    dy::T = sy / g_ny

    dt = dtCFL_kernel(params, data, ideb:ifin, dx, dy)

    if !isfinite(dt) || dt ≤ 0
        return dt  # Error handling will happen after
    elseif prev_dt == 0
        return cfl * dt
    else
        # CFL condition and maximum increase per cycle of the time step
        return convert(T, min(cfl * dt, 1.05 * prev_dt))
    end
end


function time_step(params::ArmonParameters, data::ArmonDualData)
    (; Dt, dt_on_even_cycles, cycle, cst_dt, is_root, cart_comm) = params

    params.curr_cycle_dt = params.next_cycle_dt

    if cst_dt
        params.next_cycle_dt = Dt
    elseif !dt_on_even_cycles || iseven(cycle) || params.curr_cycle_dt == 0
        @section "local_time_step" begin
            local_dt = local_time_step(params, device(data), params.curr_cycle_dt)
        end

        if params.use_MPI
            @section "time_step Allreduce" begin
                # TODO: use a non-blocking IAllreduce, which would then be probed at the end of a cycle
                #  however, we need to implement IAllreduce ourselves, since MPI.jl doesn't have a nice API for it (make a PR?)
                next_dt = MPI.Allreduce(local_dt, MPI.Op(min, data_type(params)), cart_comm)
            end
        else
            next_dt = local_dt
        end

        if (!isfinite(next_dt) || next_dt <= 0.)
            is_root && solver_error(:time, "Invalid time step for cycle $(params.cycle): $next_dt")
            return true
        end

        params.next_cycle_dt = next_dt
    else
        params.next_cycle_dt = params.curr_cycle_dt
    end

    if params.cycle == 0
        params.curr_cycle_dt = params.next_cycle_dt
    end

    return false
end


function conservation_vars_kernel(params::ArmonParameters{T, CPU_HP}, data::ArmonData, range) where T
    (; rho, Emat, domain_mask) = data
    (; dx) = params

    @batch threadlocal=zeros(T, 2) for i in range
        threadlocal[1] += rho[i]           * domain_mask[i]  # mass
        threadlocal[2] += rho[i] * Emat[i] * domain_mask[i]  # energy
    end

    threadlocal  = sum(threadlocal)  # Reduce the result of each thread

    ds = dx * dx
    total_mass   = threadlocal[1] * ds
    total_energy = threadlocal[2] * ds

    return total_mass, total_energy
end


@inline @fast function conservation_vars_kernel_reduction(rho::T, E::T, mask::T) where T
    return (
        rho * mask,     # Mass
        rho * E * mask  # Energy
    )
end


function conservation_vars_kernel(params::ArmonParameters{T, <:GPU}, data::ArmonData, range) where T
    (; rho, Emat, domain_mask) = data
    (; dx) = params

    total_mass, total_energy = @inbounds reduce(.+, @views(
        conservation_vars_kernel_reduction.(rho[range], Emat[range], domain_mask[range])
    ); init=(zero(T), zero(T)))

    ds = dx * dx
    total_mass *= ds
    total_energy *= ds

    return total_mass, total_energy
end


function conservation_vars(params::ArmonParameters{T}, data::ArmonDualData) where T
    (; ideb, ifin) = params

    total_mass, total_energy = conservation_vars_kernel(params, device(data), ideb:ifin)

    if params.use_MPI
        total_mass   = MPI.Allreduce(total_mass,   MPI.SUM, params.cart_comm)
        total_energy = MPI.Allreduce(total_energy, MPI.SUM, params.cart_comm)
    end

    return total_mass, total_energy
end
