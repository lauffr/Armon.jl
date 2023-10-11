
struct SolverStats
    final_time::Float64
    last_dt::Float64
    cycles::Int
    giga_cells_per_sec::Float64
    data::Union{Nothing, ArmonDualData}
    timer::Union{Nothing, TimerOutput}
end


function Base.show(io::IO, stats::SolverStats)
    println(io, "Solver stats:")
    println(io, " - final time:  ", @sprintf("%.18f", stats.final_time), " sec")
    println(io, " - last Δt:     ", @sprintf("%.18f", stats.last_dt), " sec")
    println(io, " - cycles:      ", stats.cycles)
    println(io, " - performance: ", round(stats.giga_cells_per_sec * 1e3, digits=3), " ×10⁶ cells-cycles/sec")
    if !isnothing(stats.timer)
        println(io, "Steps time breakdown:")
        show(io, stats.timer; compact=false, allocations=true, sortby=:firstexec)
    end
end


macro checkpoint(step_label)
    esc(:(step_checkpoint(params, data, string($step_label))))
end


function solver_cycle(params::ArmonParameters, data::ArmonDualData)
    if params.cycle == 0
        step_checkpoint(params, data, "init_test") && return true
        @section "EOS_init" update_EOS!(params, data, :full)
        step_checkpoint(params, data, "EOS_init") && return true
    end

    if params.async_comms && !haskey(params.tasks_storage, :lb)
        params.tasks_storage[:lb] = nothing
        params.tasks_storage[:rt] = nothing
        params.tasks_storage[:inner] = nothing
    end

    (@section "time_step" time_step(params, data)) && return true
    @checkpoint("time_step") && return true

    @section "$axis" for (axis, dt_factor) in split_axes(params)
        update_axis_parameters(params, axis)
        update_steps_ranges(params)
        params.cycle_dt = params.curr_cycle_dt * dt_factor

        @section "EOS" update_EOS!(params, data, :full)
        @checkpoint("EOS") && return true

        if params.async_comms
            #=
            ┌───┬────┬───────────┬────┬───┐
            │   │    │           │    │   │
            │ G │ LB │   inner   │ RT │ G │
            │   │    │           │    │   │
            └───┴────┴───────────┴────┴───┘

            - G: ghost cells
            - LB: left (X axis pass) or bottom (Y axis pass) domain
            - RT: right (X axis pass) or top (Y axis pass) domain
            - inner: inner domain

            Task graph:
                     ┌──────────────────┐
                   ┌─┤   Fluxes inner   ├─┐
                   │ └──────────────────┘ │
                   │                      │
             ┌───┐ │ ┌─────┐  ┌─────────┐ │
            ─┤EOS├─┼─┤BC LB├──┤Fluxes LB├─┼─
             └───┘ │ └─────┘  └─────────┘ │
                   │                      │
                   │ ┌─────┐  ┌─────────┐ │
                   └─┤BC RT├──┤Fluxes RT├─┘
                     └─────┘  └─────────┘

            TODO: allow async EOS by computing inner cells EOS variables in `Fluxes LB/RT` and outer
              cells EOS variables in `Fluxes inner`, breaking the mutual dependencies of those tasks
            =#

            wait(params)

            @sync begin
                @reuse_tls params.tasks_storage[:lb] @async begin
                    @section "BC lb"     async=true boundary_conditions!(params, data, :outer_lb)
                    @section "fluxes lb" async=true numerical_fluxes!(params, data, :outer_lb)
                    wait(params)  # We must wait for the CUDA/HIP stream to end before ending any task
                end

                @reuse_tls params.tasks_storage[:rt] @async begin
                    @section "BC rt"     async=true boundary_conditions!(params, data, :outer_rt)
                    @section "fluxes rt" async=true numerical_fluxes!(params, data, :outer_rt)
                    wait(params)
                end

                @reuse_tls params.tasks_storage[:inner] @async begin
                    @section "fluxes"    async=true numerical_fluxes!(params, data, :inner)
                    wait(params)
                end
            end

            @checkpoint("numerical_fluxes") && return true
        else
            @section "BC" boundary_conditions!(params, data)
            @checkpoint("boundary_conditions") && return true

            @section "fluxes" numerical_fluxes!(params, data, :full)
            @checkpoint("numerical_fluxes") && return true
        end

        @section "update" cell_update!(params, data)
        @checkpoint("cell_update") && return true

        @section "remap" projection_remap!(params, data)
        @checkpoint("projection_remap") && return true
    end

    return false
end


function time_loop(params::ArmonParameters, data::ArmonDualData)
    (; maxtime, maxcycle, nx, ny, silent, animation_step, is_root, initial_mass, initial_energy) = params

    params.cycle = 0
    params.time = 0
    params.curr_cycle_dt = 0
    params.next_cycle_dt = 0
    update_axis_parameters(params, X_axis)

    total_cycles_time = 0.

    t1 = time_ns()

    # Main solver loop
    while params.time < maxtime && params.cycle < maxcycle
        cycle_start = time_ns()

        stop = @section "solver_cycle" solver_cycle(params, data)
        stop && break

        total_cycles_time += time_ns() - cycle_start

        if is_root
            if silent <= 1
                wait(params)
                current_mass, current_energy = conservation_vars(params, data)
                ΔM = abs(initial_mass - current_mass)     / initial_mass   * 100
                ΔE = abs(initial_energy - current_energy) / initial_energy * 100
                @printf("Cycle %4d: dt = %.18f, t = %.18f, |ΔM| = %#8.6g%%, |ΔE| = %#8.6g%%\n",
                    params.cycle + 1, params.curr_cycle_dt, params.time, ΔM, ΔE)
            end
        elseif silent <= 1
            wait(params)
            conservation_vars(params, data)
        end

        params.cycle += 1
        params.time += params.curr_cycle_dt

        if animation_step != 0 && (params.cycle - 1) % animation_step == 0
            wait(params)
            frame_index = (params.cycle - 1) ÷ animation_step
            frame_file = joinpath("anim", params.output_file) * "_" * @sprintf("%03d", frame_index)
            write_sub_domain_file(params, data, frame_file)
        end
    end

    @section "Last fence" wait(params)

    @label stop

    t2 = time_ns()

    solve_time = t2 - t1
    grind_time = solve_time / (params.cycle * nx * ny)

    if is_root
        if silent < 3
            println(" ")
            println("Total time:  ", round(solve_time / 1e9, digits=5),         " sec")
            println("Cycles time: ", round(total_cycles_time / 1e9, digits=5), " sec")
            println("Grind time:  ", round(grind_time / 1e3, digits=5),        " µs/cell/cycle")
            println("Cells/sec:   ", round(1 / grind_time * 1e3, digits=5),    " Mega cells/sec")
            println("Cycles:      ", params.cycle)
            println("Last cycle:  ", 
                @sprintf("%.18f", params.time), " sec, Δt=", 
                @sprintf("%.18f", params.next_cycle_dt), " sec")
        end
    end

    return params.next_cycle_dt, params.cycle, 1 / grind_time
end

#
# Main function
#

function armon(params::ArmonParameters{T}) where T
    (; silent, is_root, timer) = params

    if is_root && silent < 3 && !isinteractive()
        print_parameters(params)
    end

    if params.use_MPI && silent < 3
        (; rank, proc_size, cart_coords) = params

        # Local info
        node_local_comm = MPI.Comm_split_type(params.global_comm, MPI.COMM_TYPE_SHARED, rank)
        local_rank = MPI.Comm_rank(node_local_comm)
        local_size = MPI.Comm_size(node_local_comm)

        is_root && println("\nProcesses info:")
        rank > 0 && MPI.Recv(Bool, rank-1, 1, params.global_comm)
        @printf(" - %2d/%-2d, local: %2d/%-2d, coords: (%2d,%-2d), cores: %3d to %3d\n", 
            rank, proc_size, local_rank, local_size, cart_coords[1], cart_coords[2], 
            minimum(getcpuids()), maximum(getcpuids()))
        rank < proc_size-1 && MPI.Send(true, rank+1, 1, params.global_comm)
    end

    if is_root && params.animation_step != 0
        if isdir("anim")
            rm.("anim/" .* readdir("anim"))
        else
            mkdir("anim")
        end
    end

    if params.measure_time
        reset_timer!(timer)
        enable_timer!(timer)
    end

    # Allocate without initialisation in order to correctly map the NUMA space using the first-touch
    # policy when working on CPU only
    @section "init" begin
        data = @section "alloc" ArmonDualData(params)
        @section "init_test" begin
            init_test(params, data)
            wait(params)
        end
    end

    if params.check_result || params.silent <= 1
        @section "Conservation variables" begin
            params.initial_mass, params.initial_energy = conservation_vars(params, data) 
        end
    end

    dt, cycles, cells_per_sec = time_loop(params, data)

    if params.check_result && is_conservative(params.test)
        @section "Conservation variables" begin
            final_mass, final_energy = conservation_vars(params, data) 
        end

        if params.is_root
            Δm = abs(final_mass   - params.initial_mass)   / params.initial_mass
            Δe = abs(final_energy - params.initial_energy) / params.initial_energy

            # Scale the tolerance with the progress in the default test case, therefore roughly
            # accounting for the number of cells (more cells -> slower time step -> more precision).
            rtol = 1e-2 * min(1, params.time / default_max_time(params.test))

            # 1% of relative error, or 10⁻¹¹ of absolute error, whichever is greater.
            Δm_ok = isapprox(Δm, 0; atol=1e-12, rtol)
            Δe_ok = isapprox(Δe, 0; atol=1e-12, rtol)

            if !(Δm_ok && Δe_ok)
                @warn "Mass and energy are not constant, the solution might not be valid!\n\
                    |mₑ-mᵢ|/mᵢ = $(@sprintf("%#8.6g", Δm))\n\
                    |Eₑ-Eᵢ|/Eᵢ = $(@sprintf("%#8.6g", Δe))\n"
            end
        end
    end

    if params.measure_time
        disable_timer!(timer)
    end

    stats = SolverStats(
        params.time, dt, cycles, cells_per_sec,
        params.return_data ? data : nothing,
        params.measure_time ? copy(timer) : nothing
    )

    if params.return_data || params.write_output || params.write_slices
        device_to_host!(data)  # No-op if the host is the device
    end

    params.write_output && write_sub_domain_file(params, data, params.output_file)
    params.write_slices && write_slices_files(params, data, params.output_file)

    if params.measure_time && params.silent < 3 && !isinteractive()
        show(params.timer)
        println()
    end

    return stats
end
