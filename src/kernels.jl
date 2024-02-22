
# TODO: make the stride deductible at compile-time (i.e. one kernel instanciation per axis)

@kernel_function function acoustic_Godunov(
    ρᵢ::T, ρᵢ₋₁::T, cᵢ::T, cᵢ₋₁::T,
    uᵢ::T, uᵢ₋₁::T, pᵢ::T, pᵢ₋₁::T
) where T
    rc_l = ρᵢ₋₁ * cᵢ₋₁
    rc_r = ρᵢ   * cᵢ
    uˢᵢ = (rc_l * uᵢ₋₁ + rc_r * uᵢ +               (pᵢ₋₁ - pᵢ)) / (rc_l + rc_r)
    pˢᵢ = (rc_r * pᵢ₋₁ + rc_l * pᵢ + rc_l * rc_r * (uᵢ₋₁ - uᵢ)) / (rc_l + rc_r)
    return uˢᵢ, pˢᵢ
end


@generic_kernel function acoustic!(
    s::Int, uˢ_::V, pˢ_::V,
    ρ::V, uₐ::V, p::V, c::V
) where V
    u = uₐ  # `u` or `v` depending on the current axis
    i = @index_2D_lin()
    uˢ_[i], pˢ_[i] = acoustic_Godunov(
        ρ[i], ρ[i-s], c[i], c[i-s],
        u[i], u[i-s], p[i], p[i-s]
    )
end


@generic_kernel function acoustic_GAD!(
    s::Int, dt::T, dx::T,
    uˢ::V, pˢ::V, ρ::V, uₐ::V, p::V, c::V,
    limiter_tag::LimiterType
) where {T, V <: AbstractArray{T}, LimiterType <: Limiter}
    i = @index_2D_lin()

    u = uₐ  # `u` or `v` depending on the current axis

    # First order acoustic solver on the left cell
    uˢ_i₋, pˢ_i₋ = acoustic_Godunov(
        ρ[i-s], ρ[i-2s], c[i-s], c[i-2s],
        u[i-s], u[i-2s], p[i-s], p[i-2s]
    )

    # First order acoustic solver on the current cell
    uˢ_i, pˢ_i = acoustic_Godunov(
        ρ[i], ρ[i-s], c[i], c[i-s],
        u[i], u[i-s], p[i], p[i-s]
    )

    # First order acoustic solver on the right cell
    uˢ_i₊, pˢ_i₊ = acoustic_Godunov(
        ρ[i+s], ρ[i], c[i+s], c[i],
        u[i+s], u[i], p[i+s], p[i]
    )

    # Second order GAD acoustic solver on the current cell

    r_u₋ = (uˢ_i₊  -  u[i]) / (uˢ_i - u[i-s] + T(1e-6))
    r_p₋ = (pˢ_i₊  -  p[i]) / (pˢ_i - p[i-s] + T(1e-6))
    r_u₊ = (u[i-s] - uˢ_i₋) / (u[i] - uˢ_i   + T(1e-6))
    r_p₊ = (p[i-s] - pˢ_i₋) / (p[i] - pˢ_i   + T(1e-6))

    r_u₋ = limiter(r_u₋, LimiterType())
    r_p₋ = limiter(r_p₋, LimiterType())
    r_u₊ = limiter(r_u₊, LimiterType())
    r_p₊ = limiter(r_p₊, LimiterType())

    dm_l = ρ[i-s] * dx
    dm_r = ρ[i]   * dx
    Dm   = (dm_l + dm_r) / 2

    rc_l = ρ[i-s] * c[i-s]
    rc_r = ρ[i]   * c[i]
    θ    = T(0.5) * (1 - (rc_l + rc_r) / 2 * (dt / Dm))

    uˢ[i] = uˢ_i + θ * (r_u₊ * (u[i] - uˢ_i) - r_u₋ * (uˢ_i - u[i-s]))
    pˢ[i] = pˢ_i + θ * (r_p₊ * (p[i] - pˢ_i) - r_p₋ * (pˢ_i - p[i-s]))
end


@generic_kernel function perfect_gas_EOS!(
    γ::T,
    ρ::V, E::V, u::V, v::V, p::V, c::V, g::V
) where {T, V <: AbstractArray{T}}
    i = @index_2D_lin()
    e = E[i] - 0.5 * (u[i]^2 + v[i]^2)
    p[i] = (γ - 1.) * ρ[i] * e
    c[i] = sqrt(γ * p[i] / ρ[i])
    g[i] = (1. + γ) / 2
end


@generic_kernel function bizarrium_EOS!(
    ρ::V, u::V, v::V, E::V, p::V, c::V, g::V
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

    x = ρ[i] / rho0 - 1
    G = G0 * (1-rho0 / ρ[i])

    f0 = (1+(s/3-2)*x+q*x^2+r*x^3)/(1-s*x)
    f1 = (s/3-2+2*q*x+3*r*x^2+s*f0)/(1-s*x)
    f2 = (2*q+6*r*x+2*s*f1)/(1-s*x)
    f3 = (6*r+3*s*f2)/(1-s*x)

    epsk0     = eps0 - Cv0*T0*(1+G) + 0.5*(K0/rho0)*x^2*f0
    pk0       = -Cv0*T0*G0*rho0 + 0.5*K0*x*(1+x)^2*(2*f0+x*f1)
    pk0prime  = -0.5*K0*(1+x)^3*rho0 * (2*(1+3x)*f0 + 2*x*(2+3x)*f1 + x^2*(1+x)*f2)
    pk0second = 0.5*K0*(1+x)^4*rho0^2 * (12*(1+2x)*f0 + 6*(1+6x+6*x^2)*f1 + 
                                                    6*x*(1+x)*(1+2x)*f2 + x^2*(1+x)^2*f3)

    e = E[i] - 0.5 * (u[i]^2 + v[i]^2)
    p[i] = pk0 + G0 * rho0 * (e - epsk0)
    c[i] = sqrt(G0 * rho0 * (p[i] - pk0) - pk0prime) / ρ[i]
    g[i] = 0.5 / (ρ[i]^3 * c[i]^2) * (pk0second + (G0 * rho0)^2 * (p[i] - pk0))
end


@generic_kernel function cell_update!(
    s::Int, dx::T, dt::T, 
    uˢ::V, pˢ::V, ρ::V, uₐ::V, E::V
) where {T, V <: AbstractArray{T}}
    i = @index_2D_lin()
    u = uₐ  # `u` or `v` depending on the current axis
    dm = ρ[i] * dx
    ρ[i]  = dm / (dx + dt * (uˢ[i+s] - uˢ[i]))
    u[i] += dt / dm * (pˢ[i]         - pˢ[i+s]          )
    E[i] += dt / dm * (pˢ[i] * uˢ[i] - pˢ[i+s] * uˢ[i+s])
end


@generic_kernel function euler_projection!(
    s::Int, dx::T, dt::T,
    uˢ::V, ρ::V, u::V, v::V, E::V,
    advection_ρ::V, advection_uρ::V, advection_vρ::V, advection_Eρ::V
) where {T, V <: AbstractArray{T}}
    i = @index_2D_lin()

    dX = dx + dt * (uˢ[i+s] - uˢ[i])

    tmp_ρ  = (dX * ρ[i]        - (advection_ρ[i+s]  - advection_ρ[i] )) / dx
    tmp_uρ = (dX * ρ[i] * u[i] - (advection_uρ[i+s] - advection_uρ[i])) / dx
    tmp_vρ = (dX * ρ[i] * v[i] - (advection_vρ[i+s] - advection_vρ[i])) / dx
    tmp_Eρ = (dX * ρ[i] * E[i] - (advection_Eρ[i+s] - advection_Eρ[i])) / dx

    ρ[i] = tmp_ρ
    u[i] = tmp_uρ / tmp_ρ
    v[i] = tmp_vρ / tmp_ρ
    E[i] = tmp_Eρ / tmp_ρ
end


@generic_kernel function advection_first_order!(
    s::Int, dt::T,
    uˢ::V, ρ::V, u::V, v::V, E::V,
    advection_ρ::V, advection_uρ::V, advection_vρ::V, advection_Eρ::V
) where {T, V <: AbstractArray{T}}
    i = @index_2D_lin()
    is = i
    disp = dt * uˢ[i]
    if disp > 0
        i = i - s
    end

    advection_ρ[is]  = disp * (ρ[i]       )
    advection_uρ[is] = disp * (ρ[i] * u[i])
    advection_vρ[is] = disp * (ρ[i] * v[i])
    advection_Eρ[is] = disp * (ρ[i] * E[i])
end


@generic_kernel function advection_second_order!(
    s::Int, dx::T, dt::T,
    uˢ::V, ρ::V, u::V, v::V, E::V,
    advection_ρ::V, advection_uρ::V, advection_vρ::V, advection_Eρ::V
) where {T, V <: AbstractArray{T}}
    i = @index_2D_lin()
    is = i
    disp = dt * uˢ[i]
    if disp > 0
        Δxₑ = -(dx - dt * uˢ[i-s])
        i = i - s
    else
        Δxₑ = dx + dt * uˢ[i+s]
    end

    Δxₗ₋  = dx + dt * (uˢ[i]    - uˢ[i-s])
    Δxₗ   = dx + dt * (uˢ[i+s]  - uˢ[i]  )
    Δxₗ₊  = dx + dt * (uˢ[i+2s] - uˢ[i+s])

    r₋  = (2 * Δxₗ) / (Δxₗ + Δxₗ₋)
    r₊  = (2 * Δxₗ) / (Δxₗ + Δxₗ₊)

    slopes_ρ  = slope_minmod(ρ[i-s]         , ρ[i]       , ρ[i+s]         , r₋, r₊)
    slopes_uρ = slope_minmod(ρ[i-s] * u[i-s], ρ[i] * u[i], ρ[i+s] * u[i+s], r₋, r₊)
    slopes_vρ = slope_minmod(ρ[i-s] * v[i-s], ρ[i] * v[i], ρ[i+s] * v[i+s], r₋, r₊)
    slopes_Eρ = slope_minmod(ρ[i-s] * E[i-s], ρ[i] * E[i], ρ[i+s] * E[i+s], r₋, r₊)

    length_factor = Δxₑ / (2 * Δxₗ)
    advection_ρ[is]  = disp * (ρ[i]        - slopes_ρ  * length_factor)
    advection_uρ[is] = disp * (ρ[i] * u[i] - slopes_uρ * length_factor)
    advection_vρ[is] = disp * (ρ[i] * v[i] - slopes_vρ * length_factor)
    advection_Eρ[is] = disp * (ρ[i] * E[i] - slopes_Eρ * length_factor)
end


@generic_kernel function init_test(
    global_grid::NTuple{2, Int},
    cart_coords::NTuple{2, Int}, sub_domain_size::NTuple{2, Int},
    block_pos::NTuple{2, Int}, static_bsize::NTuple{2, Int}, bsize::BSize,
    domain_size::NTuple{2, T}, origin::NTuple{2, T},
    x::V, y::V, ρ::V, E::V, u::V, v::V,
    mask::V, p::V, c::V, g::V, uˢ::V, pˢ::V,
    test_case::Test, debug_indexes::Bool
) where {T, V <: AbstractArray{T}, Test <: TestCase, BSize <: BlockSize}
    @kernel_init begin
        # Position of the origin of this block
        pos = cart_coords .* sub_domain_size .+ (block_pos .- 1) .* static_bsize

        if Test <: TwoStateTestCase
            (; high_ρ::T, low_ρ::T,
               high_E::T, low_E::T,
               high_u::T, low_u::T,
               high_v::T, low_v::T) = init_test_params(test_case, T)
        end
    end

    i = @index_2D_lin()
    I = position(bsize, i)  # Position in the block's real cells

    # Position in the global grid
    (x[i], y[i]) = (I .+ pos) ./ global_grid .* domain_size .+ origin

    # Middle point of the cell
    mid = (x[i], y[i]) .+ domain_size ./ (2 .* global_grid)

    if debug_indexes
        global_i = sum((I .+ pos .- 1) .* Base.size_to_strides(1, sub_domain_size...)) + 1
        ρ[i] = global_i
        E[i] = global_i
        u[i] = global_i
        v[i] = global_i
    else
        if Test <: TwoStateTestCase
            if test_region_high(mid, test_case)
                ρ[i] = high_ρ
                E[i] = high_E
                u[i] = high_u
                v[i] = high_v
            else
                ρ[i] = low_ρ
                E[i] = low_E
                u[i] = low_u
                v[i] = low_v
            end
        else
            init_vals = init_test_params(mid, test_case)::InitTestParams{T}
            ρ[i] = init_vals.ρ
            E[i] = init_vals.E
            u[i] = init_vals.u
            v[i] = init_vals.v
        end
    end

    # Set the domain mask to 1 if the cell is real or 0 otherwise
    mask[i] = !Bool(is_ghost(bsize, i))

    # TODO: remove this as it should be unnecessary now
    # Set to zero to make sure no non-initialized values changes the result
    p[i] = 0
    c[i] = 1  # Set to 1 as a max speed of 0 will create NaNs
    g[i] = 0
    uˢ[i] = 0
    pˢ[i] = 0
end

#
# Wrappers
#

function numerical_fluxes!(params::ArmonParameters, blk::LocalTaskBlock)
    # TODO: use methods instead of Symbols
    range = block_domain_range(blk.size, params.steps_ranges.fluxes)
    dt = params.cycle_dt
    u = params.current_axis == X_axis ? blk.u : blk.v
    s = stride_along(blk.size, params.current_axis)
    if params.riemann == :acoustic  # 2-state acoustic solver (Godunov)
        if params.scheme == :Godunov
            return acoustic!(params, blk, range, s, blk.uˢ, blk.pˢ, u)
        elseif params.scheme == :GAD
            return acoustic_GAD!(params, blk, range, s, dt, u, params.riemann_limiter)
        else
            solver_error(:config, "Unknown acoustic scheme: ", params.scheme)
        end
    else
        solver_error(:config, "Unknown Riemann solver: ", params.riemann)
    end
end


function numerical_fluxes!(params::ArmonParameters, grid::BlockGrid)
    @iter_blocks for blk in device_blocks(grid)
        numerical_fluxes!(params, blk)
    end
end


function update_EOS!(params::ArmonParameters, blk::LocalTaskBlock, t::TestCase)
    range = block_domain_range(blk.size, params.steps_ranges.EOS)
    gamma = data_type(params)(specific_heat_ratio(t))
    return perfect_gas_EOS!(params, blk, range, gamma)
end


function update_EOS!(params::ArmonParameters, blk::LocalTaskBlock, ::Bizarrium)
    range = block_domain_range(blk.size, params.steps_ranges.EOS)
    return bizarrium_EOS!(params, blk, range)
end


function update_EOS!(params::ArmonParameters, grid::BlockGrid)
    @iter_blocks for blk in device_blocks(grid)
        update_EOS!(params, blk, params.test)
    end
end


function init_test(params::ArmonParameters, blk::LocalTaskBlock)
    blk_domain = block_domain_range(blk.size, params.steps_ranges.full_domain)
    real_static_bsize = params.block_size .- 2*params.nghost
    init_test(params, blk, blk_domain, Tuple(blk.pos), real_static_bsize, blk.size, params.test)
end


function init_test(params::ArmonParameters, grid::BlockGrid)
    @iter_blocks for blk in device_blocks(grid)
        init_test(params, blk)
    end
end


function cell_update!(params::ArmonParameters, blk::LocalTaskBlock)
    blk_domain = block_domain_range(blk.size, params.steps_ranges.cell_update)
    u = params.current_axis == X_axis ? blk.u : blk.v
    s = stride_along(blk.size, params.current_axis)
    cell_update!(params, blk, blk_domain, s, params.cycle_dt, u)
end


function cell_update!(params::ArmonParameters, grid::BlockGrid)
    @iter_blocks for blk in device_blocks(grid)
        cell_update!(params, blk)
    end
end


@kernel_function function slope_minmod(uᵢ₋::T, uᵢ::T, uᵢ₊::T, r₋::T, r₊::T) where T
    Δu₊ = r₊ * (uᵢ₊ - uᵢ )
    Δu₋ = r₋ * (uᵢ  - uᵢ₋)
    s = sign(Δu₊)
    return s * max(0, min(s * Δu₊, s * Δu₋))
end


function advection_first_order!(params::ArmonParameters, blk::LocalTaskBlock)
    advection_range = block_domain_range(blk.size, params.steps_ranges.advection)
    s = stride_along(blk.size, params.current_axis)
    advection_first_order!(params, blk, advection_range, s, params.cycle_dt,
        blk.work_1, blk.work_2, blk.work_3, blk.work_4)
end


function advection_second_order!(params::ArmonParameters, blk::LocalTaskBlock)
    advection_range = block_domain_range(blk.size, params.steps_ranges.advection)
    s = stride_along(blk.size, params.current_axis)
    advection_second_order!(params, blk, advection_range, s, params.cycle_dt,
        blk.work_1, blk.work_2, blk.work_3, blk.work_4)
end


function euler_projection!(params::ArmonParameters, blk::LocalTaskBlock)
    projection_range = block_domain_range(blk.size, params.steps_ranges.projection)
    s = stride_along(blk.size, params.current_axis)
    euler_projection!(params, blk, projection_range, s, params.cycle_dt,
        blk.work_1, blk.work_2, blk.work_3, blk.work_4)
end


function projection_remap!(params::ArmonParameters, blk::LocalTaskBlock)
    if params.projection == :euler
        advection_first_order!(params, blk)
    elseif params.projection == :euler_2nd
        advection_second_order!(params, blk)
    else
        solver_error(:config, "Unknown projection scheme: $(params.projection)")
    end

    return euler_projection!(params, blk)
end


function projection_remap!(params::ArmonParameters, grid::BlockGrid)
    @section "Advection" if params.projection == :euler
        @iter_blocks for blk in device_blocks(grid)
            advection_first_order!(params, blk)
        end
    elseif params.projection == :euler_2nd
        @iter_blocks for blk in device_blocks(grid)
            advection_second_order!(params, blk)
        end
    else
        solver_error(:config, "Unknown projection scheme: $(params.projection)")
    end

    @section "Projection" @iter_blocks for blk in device_blocks(grid)
        euler_projection!(params, blk)
    end
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


@inline @fast function dtCFL_kernel_reduction(u::T, v::T, c::T, dx::T, dy::T) where T
    # Mask-less version
    return min(
        dx / abs(max(abs(u + c), abs(u - c))),
        dy / abs(max(abs(v + c), abs(v - c)))
    )
end


@fast function dtCFL_kernel(params::ArmonParameters{T, CPU_HP}, blk::LocalTaskBlock, dx, dy) where T
    # Reduction exploiting multithreading from the caller
    (; u, v, c) = blk
    range = block_domain_range(blk.size, params.steps_ranges.real_domain)

    res = typemax(T)
    for j in range.col, i in range.row .+ (j - 1)
        cell_dt = dtCFL_kernel_reduction(u[i], v[i], c[i], dx, dy)
        res = min(res, cell_dt)
    end
    return res
end


@generic_kernel function dtCFL_kernel(
    u::V, v::V, c::V, res::V, bsize::BlockSize, dx::T, dy::T
) where {T, V <: AbstractArray{T}}
    i = @index_2D_lin()
    mask = is_ghost(bsize, i)
    res[i] = dtCFL_kernel_reduction(u[i], v[i], c[i], mask, dx, dy)
end


function dtCFL_kernel(params::ArmonParameters, blk::LocalTaskBlock, dx, dy)
    range = block_domain_range(blk.size, params.steps_ranges.full_domain)
    if params.use_two_step_reduction
        # Use a temporary array to store the partial reduction result. This is inefficient but can
        # be more performant for some GPU backends.
        # We avoid filling the whole array with `typemax(T)` by applying the kernel on the whole
        # array (`full_domain`) and by using `is_ghost(i)` as a mask.
        # TODO: There may be some synchronization issues with oneAPI.jl.
        dtCFL_kernel(params, blk, range, blk.work_1, blk.size, dx, dy)
        wait(params)
        return reduce(min, blk.work_1)
    else
        # Direct reduction, which depends on a pre-computed `mask`
        lin_range = first(range):last(range)  # Reduce on a 1D range
        c_v    = @view blk.c[lin_range]
        u_v    = @view blk.u[lin_range]
        v_v    = @view blk.v[lin_range]
        mask_v = @view blk.mask[lin_range]
        return mapreduce(dtCFL_kernel_reduction, min, u_v, v_v, c_v, mask_v, dx, dy)
    end
end


function dtCFL_kernel(params::ArmonParameters{T}, grid::BlockGrid, dx, dy) where T
    threads_res = Vector{T}(undef, params.use_threading ? Threads.nthreads() : 1)
    threads_res .= typemax(T)

    @iter_blocks for blk in device_blocks(grid)
        blk_res = dtCFL_kernel(params, blk, dx, dy)

        tid = Threads.threadid()
        threads_res[tid] = min(blk_res, threads_res[tid])
    end

    return minimum(threads_res)
end


function local_time_step(params::ArmonParameters{T}, grid::BlockGrid) where {T}
    (; cfl, global_grid, domain_size) = params

    (dx::T, dy::T) = domain_size ./ global_grid

    # Time step for this sub-domain
    dt = dtCFL_kernel(params, grid, dx, dy)

    prev_dt = params.curr_cycle_dt

    if !isfinite(dt) || dt ≤ 0
        return dt  # Error handling will happen afterwards
    elseif prev_dt == 0
        return cfl * dt
    else
        # CFL condition and maximum increase per cycle of the time step
        return convert(T, min(cfl * dt, 1.05 * prev_dt))
    end
end


function time_step(params::ArmonParameters, grid::BlockGrid)
    (; Dt, dt_on_even_cycles, cycle, cst_dt, is_root, cart_comm) = params

    params.curr_cycle_dt = params.next_cycle_dt

    if cst_dt
        params.next_cycle_dt = Dt
    elseif !dt_on_even_cycles || iseven(cycle) || params.curr_cycle_dt == 0
        @section "local_time_step" begin
            local_dt = local_time_step(params, grid)
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


@inline @fast function conservation_vars_kernel_reduction(ρ::T, E::T, mask::T) where T
    return (
        ρ * mask,     # Mass
        ρ * E * mask  # Energy
    )
end


@inline @fast function conservation_vars_kernel_reduction(ρ::T, E::T) where T
    # Mask-less version
    return (
        ρ,     # Mass
        ρ * E  # Energy
    )
end


@fast function conservation_vars(params::ArmonParameters{T, CPU_HP}, blk::LocalTaskBlock) where T
    # Reduction exploiting multithreading from the caller
    (; ρ, E) = blk
    range = block_domain_range(blk.size, params.steps_ranges.real_domain)

    res_mass = zero(T)
    res_energy = zero(T)
    for j in range.col, i in range.row .+ (j - 1)
        (res_mass, res_energy) = (res_mass, res_energy) .+ conservation_vars_kernel_reduction(ρ[i], E[i])
    end

    ds = params.dx^2
    res_mass   *= ds
    res_energy *= ds

    return res_mass, res_energy
end


@generic_kernel function conservation_vars(
    ρ::V, E::V, res_mass::V, res_energy::V, bsize::BlockSize
) where {V}
    i = @index_2D_lin()
    mask = is_ghost(bsize, i)
    (res_mass[i], res_energy[i]) = conservation_vars_kernel_reduction(ρ[i], E[i], mask)
end


function conservation_vars(params::ArmonParameters{T}, blk::LocalTaskBlock) where {T}
    range = block_domain_range(blk.size, params.steps_ranges.full_domain)
    if params.use_two_step_reduction
        # Use a temporary array to store the partial reduction results.
        # Same comments as for `dtCFL_kernel`.
        conservation_vars(params, blk, range, blk.work_1, blk.work_2, blk.size)
        wait(params)
        identity2(a, b) = (a, b)
        total_mass, total_energy = mapreduce(identity2, min, blk.work_1, blk.work_2;
            init=(zero(T), zero(T)))
    else
        lin_range = first(range):last(range)
        ρ_v    = @view blk.ρ[lin_range]
        E_v    = @view blk.E[lin_range]
        mask_v = @view blk.mask[lin_range]
        total_mass, total_energy = mapreduce(conservation_vars_kernel_reduction, .+, ρ_v, E_v, mask_v;
            init=(zero(T), zero(T)))
    end

    ds = params.dx^2
    total_mass   *= ds
    total_energy *= ds

    return total_mass, total_energy
end


function conservation_vars(params::ArmonParameters{T}, grid::BlockGrid) where {T}
    threads_mass   = Vector{T}(undef, params.use_threading ? Threads.nthreads() : 1)
    threads_energy = Vector{T}(undef, params.use_threading ? Threads.nthreads() : 1)
    threads_mass   .= 0
    threads_energy .= 0

    @iter_blocks for blk in device_blocks(grid)
        tid = Threads.threadid()
        (threads_mass[tid], threads_energy[tid]) = (threads_mass[tid], threads_energy[tid]) .+ conservation_vars(params, blk)
    end

    total_mass   = sum(threads_mass)
    total_energy = sum(threads_energy)

    if params.use_MPI
        total_mass   = MPI.Allreduce(total_mass,   MPI.SUM, params.cart_comm)
        total_energy = MPI.Allreduce(total_energy, MPI.SUM, params.cart_comm)
    end

    return total_mass, total_energy
end
