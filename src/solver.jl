
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
    esc(:(step_checkpoint(params, data, string($step_label); dependencies)))
end


function init_time_step(params::ArmonParameters, data::ArmonDualData)
    if params.Dt == 0
        # No imposed initial time step, we must compute the first one manually
        update_EOS!(params, data, :full)
        step_checkpoint(params, data, "update_EOS_init") && return true
        time_step(params, data)
        params.curr_cycle_dt = params.next_cycle_dt
    else
        params.next_cycle_dt = Dt
        params.curr_cycle_dt = Dt
    end

    return false
end


function solver_cycle(params::ArmonParameters, data::ArmonDualData; dependencies=NoneEvent())
    (; timer) = params
    @timeit timer "time_step" time_step(params, data; dependencies)

    for (axis, dt_factor) in split_axes(params)
        update_axis_parameters(params, axis)
        update_steps_ranges(params)
        params.cycle_dt = params.curr_cycle_dt * dt_factor

        @timeit timer "$axis" begin

        if params.async_comms
            dependencies = @timeit timer "EOS lb" update_EOS!(params, data, :outer_lb; dependencies)
            dependencies = @timeit timer "EOS rt" update_EOS!(params, data, :outer_rt; dependencies)

            outer_lb_BC = @timeit timer "BC lb" boundaryConditions!(params, data, :outer_lb; dependencies)
            outer_rt_BC = @timeit timer "BC rt" boundaryConditions!(params, data, :outer_rt; dependencies)

            dependencies = @timeit timer "EOS" update_EOS!(params, data, :inner; dependencies)
            dependencies = @timeit timer "fluxes" numericalFluxes!(params, data, :inner; dependencies)

            dependencies = MultiEvent((dependencies, outer_lb_BC, outer_rt_BC))
            dependencies = @timeit timer "Post BC" post_boundary_conditions(params, data; dependencies)

            dependencies = @timeit timer "fluxes lb" numericalFluxes!(params, data, :outer_lb; dependencies)
            dependencies = @timeit timer "fluxes rt" numericalFluxes!(params, data, :outer_rt; dependencies)

            # TODO: async cellUpdate?
        else
            dependencies = @timeit timer "EOS" update_EOS!(params, data, :full; dependencies)
            @checkpoint("update_EOS") && @goto stop

            dependencies = @timeit timer "BC" boundaryConditions!(params, data; dependencies)
            @checkpoint("boundaryConditions") && @goto stop

            dependencies = @timeit timer "fluxes" numericalFluxes!(params, data, :full; dependencies)
            @checkpoint("numericalFluxes") && @goto stop
        end

        dependencies = @timeit timer "update" cellUpdate!(params, data; dependencies)
        @checkpoint("cellUpdate") && @goto stop

        dependencies = @timeit timer "remap" projection_remap!(params, data; dependencies)
        @checkpoint("projection_remap") && @goto stop

        end
    end

    return dependencies, false

    @label stop
    return dependencies, true
end


function time_loop(params::ArmonParameters, data::ArmonDualData)
    (; maxtime, maxcycle, nx, ny, silent, animation_step, is_root) = params

    params.cycle = 0
    params.time = 0
    params.curr_cycle_dt = 0
    params.next_cycle_dt = 0

    total_cycles_time = 0.

    if silent <= 1
        initial_mass, initial_energy = conservation_vars(params, data)
    end

    t1 = time_ns()

    (@timeit params.timer "init_time_step" init_time_step(params, data)) && @goto stop

    dependencies = NoneEvent()

    # Main solver loop
    while params.time < maxtime && params.cycle < maxcycle
        cycle_start = time_ns()

        dependencies, stop = @timeit params.timer "solver_cycle" solver_cycle(params, data; dependencies)
        stop && @goto stop

        total_cycles_time += time_ns() - cycle_start

        if is_root
            if silent <= 1
                wait(params, dependencies)
                current_mass, current_energy = conservation_vars(params, data)
                ΔM = abs(initial_mass - current_mass)     / initial_mass   * 100
                ΔE = abs(initial_energy - current_energy) / initial_energy * 100
                @printf("Cycle %4d: dt = %.18f, t = %.18f, |ΔM| = %#8.6g%%, |ΔE| = %#8.6g%%\n",
                    params.cycle + 1, params.curr_cycle_dt, params.time, ΔM, ΔE)
            end
        elseif silent <= 1
            wait(params, dependencies)
            conservation_vars(params, data)
        end

        params.cycle += 1
        params.time += params.curr_cycle_dt

        if animation_step != 0 && (params.cycle - 1) % animation_step == 0
            wait(params, dependencies)
            frame_index = (params.cycle - 1) ÷ animation_step
            frame_file = joinpath("anim", params.output_file) * "_" * @sprintf("%03d", frame_index)
            write_sub_domain_file(params, data, frame_file)
        end
    end

    wait(params, dependencies)

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

    if is_root && silent < 3
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
    @timeit timer "init" begin
        data = @timeit timer "alloc" ArmonDualData(params)
        @timeit timer "init_test" wait(params, init_test(params, data))
    end

    dt, cycles, cells_per_sec = time_loop(params, data)

    if params.measure_time
        disable_timer!(timer)
    end

    stats = SolverStats(
        params.time, dt, cycles, cells_per_sec,
        params.return_data ? data : nothing,
        params.measure_time ? copy(timer) : nothing
    )

    params.use_gpu && device_to_host!(data)
    params.write_output && write_sub_domain_file(params, data, params.output_file)
    params.write_slices && write_slices_files(params, data, params.output_file)

    if params.measure_time && params.silent < 3
        show(params.timer)
        println()
    end

    return stats
end
