
macro checkpoint(step_label)
    esc(:(step_checkpoint(params, data, cpu_data, string($step_label); dependencies)))
end


function init_time_step(params::ArmonParameters, data::ArmonData, cpu_data::ArmonData)
    if params.Dt == 0
        # No imposed initial time step, we must compute the first one manually
        update_EOS!(params, data, :full) |> wait
        step_checkpoint(params, data, cpu_data, "update_EOS_init") && return true
        time_step(params, data)
        params.curr_cycle_dt = params.next_cycle_dt
    else
        params.next_cycle_dt = Dt
        params.curr_cycle_dt = Dt
    end

    return false
end


function solver_cycle(params::ArmonParameters, data::ArmonData, cpu_data::ArmonData, host_array::Vector; 
        dependencies=NoneEvent())

    # Future async structure:
    # - update_EOS outer lb/rt
    # - async BC (outer lb/rt)
    # - update_EOS inner
    # - fluxes inner
    # - join async BC
    # - fluxes outer lb/rt

    # Sync structure:
    # - update_EOS full
    # - sync BC (outer lb/rt)
    # - fluxes full

    time_step(params, data; dependencies)

    for (axis, dt_factor) in split_axes(params)
        update_axis_parameters(params, axis)
        update_steps_ranges(params)
        params.cycle_dt = params.curr_cycle_dt * dt_factor

        dependencies = update_EOS!(params, data, :full; dependencies)
        @checkpoint("update_EOS") && @goto stop

        dependencies = boundaryConditions!(params, data, host_array; dependencies)
        @checkpoint("boundaryConditions") && @goto stop

        dependencies = numericalFluxes!(params, data, :full; dependencies)
        @checkpoint("numericalFluxes") && @goto stop

        @perf_task "loop" "cellUpdate" dependencies = cellUpdate!(params, data; dependencies)
        @checkpoint("cellUpdate") && @goto stop

        @perf_task "loop" "euler_proj" dependencies = projection_remap!(params, data, host_array; dependencies)
        @checkpoint("projection_remap") && @goto stop
    end

    return dependencies, false

    @label stop
    return dependencies, true
end


function time_loop(params::ArmonParameters{T}, data::ArmonData{V},
        cpu_data::ArmonData{W}) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; maxtime, maxcycle, nx, ny, silent, animation_step, is_root) = params

    total_cycles_time::T = 0.

    if params.use_MPI && params.use_gpu
        # Host version of temporary array used for MPI communications
        host_array = Vector{T}(undef, params.comm_array_size)
    else
        host_array = Vector{T}()
    end

    if silent <= 1
        initial_mass, initial_energy = conservation_vars(params, data)
    end

    t1 = time_ns()
    t_warmup = t1

    init_time_step(params, data, cpu_data) && @goto stop

    dependencies = NoneEvent()

    # Main solver loop
    while params.time < maxtime && params.cycle < maxcycle
        cycle_start = time_ns()

        dependencies, stop = solver_cycle(params, data, cpu_data, host_array; dependencies)
        stop && @goto stop

        if !is_warming_up()
            total_cycles_time += time_ns() - cycle_start
        end

        if is_root
            if silent <= 1
                wait(dependencies)
                current_mass, current_energy = conservation_vars(params, data)
                ΔM = abs(initial_mass - current_mass)     / initial_mass   * 100
                ΔE = abs(initial_energy - current_energy) / initial_energy * 100
                @printf("Cycle %4d: dt = %.18f, t = %.18f, |ΔM| = %#8.6g%%, |ΔE| = %#8.6g%%\n",
                    params.cycle + 1, params.curr_cycle_dt, params.time, ΔM, ΔE)
            end
        elseif silent <= 1
            wait(dependencies)
            conservation_vars(params, data)
        end

        params.cycle += 1
        params.time += params.curr_cycle_dt

        if params.cycle == 5
            wait(dependencies)
            t_warmup = time_ns()
            set_warmup(false)
        end

        if animation_step != 0 && (params.cycle - 1) % animation_step == 0
            wait(dependencies)
            frame_index = (params.cycle - 1) ÷ animation_step
            frame_file = joinpath("anim", params.output_file) * "_" * @sprintf("%03d", frame_index)
            write_sub_domain_file(params, data, frame_file)
        end
    end

    wait(dependencies)

    @label stop

    t2 = time_ns()

    nb_cells = nx * ny
    grind_time = (t2 - t_warmup) / ((params.cycle - 5) * nb_cells)

    if is_root
        if params.compare
            # ignore timing errors when comparing
        elseif params.cycle <= 5 && maxcycle > 5
            error("More than 5 cycles are needed to compute the grind time, got: $(params.cycle)")
        elseif t2 < t_warmup
            error("Clock error: $t2 < $t_warmup")
        end

        if silent < 3
            println(" ")
            println("Total time:  ", round((t2 - t1) / 1e9, digits=5),         " sec")
            println("Cycles time: ", round(total_cycles_time / 1e9, digits=5), " sec")
            println("Warmup:      ", round((t_warmup - t1) / 1e9, digits=5),   " sec")
            println("Grind time:  ", round(grind_time / 1e3, digits=5),        " µs/cell/cycle")
            println("Cells/sec:   ", round(1 / grind_time * 1e3, digits=5),    " Mega cells/sec")
            println("Cycles:      ", params.cycle)
            println("Last cycle:  ", 
                @sprintf("%.18f", params.time), " sec, Δt=", 
                @sprintf("%.18f", params.next_cycle_dt), " sec")
        end
    end

    return params.next_cycle_dt, params.cycle, convert(T, 1 / grind_time), total_cycles_time
end

#
# Main function
#

function armon(params::ArmonParameters{T}) where T
    (; silent, is_root) = params

    if params.measure_time
        empty!(axis_time_contrib)
        empty!(total_time_contrib)
        set_warmup(true)
    end

    if is_root && silent < 3
        print_parameters(params)
    end

    if params.use_MPI && silent < 3
        (; rank, proc_size, cart_coords) = params

        # Local info
        node_local_comm = MPI.Comm_split_type(COMM, MPI.COMM_TYPE_SHARED, rank)
        local_rank = MPI.Comm_rank(node_local_comm)
        local_size = MPI.Comm_size(node_local_comm)

        is_root && println("\nProcesses info:")
        rank > 0 && MPI.Recv(Bool, rank-1, 1, COMM)
        @printf(" - %2d/%-2d, local: %2d/%-2d, coords: (%2d,%-2d), cores: %3d to %3d\n", 
            rank, proc_size, local_rank, local_size, cart_coords[1], cart_coords[2], 
            minimum(getcpuids()), maximum(getcpuids()))
        rank < proc_size-1 && MPI.Send(true, rank+1, 1, COMM)
    end

    if is_root && params.animation_step != 0
        if isdir("anim")
            rm.("anim/" .* readdir("anim"))
        else
            mkdir("anim")
        end
    end

    # Allocate without initialisation in order to correctly map the NUMA space using the first-touch
    # policy when working on CPU only
    @perf_task "init" "alloc" data = ArmonData(params)
    @perf_task "init" "init_test" wait(init_test(params, data))

    if params.use_gpu
        copy_time = @elapsed d_data = data_to_gpu(data, get_device_array(params))
        (is_root && silent <= 2) && @printf("Time for copy to device: %.3g sec\n", copy_time)

        @pretty_time dt, cycles, cells_per_sec, total_time = time_loop(params, d_data, data)

        data_from_gpu(data, d_data)
    else
        @pretty_time dt, cycles, cells_per_sec, total_time = time_loop(params, data, data)
    end

    if params.write_output
        write_sub_domain_file(params, data, params.output_file)
    end

    if params.write_slices
        write_slices_files(params, data, params.output_file)
    end

    sorted_time_contrib = sort(collect(total_time_contrib))

    if params.measure_time && length(sorted_time_contrib) > 0
        sync_total_time = mapreduce(x->x[2], +, sorted_time_contrib)
        async_efficiency = (sync_total_time - total_time) / total_time
        async_efficiency = max(async_efficiency, 0.)
    else
        sync_total_time = 1.
        async_efficiency = 0.
    end

    if is_root && params.measure_time && silent < 3 && !isempty(axis_time_contrib)
        axis_time = Dict{Axis, Float64}()

        # Print the time of each step for each axis
        for (axis, time_contrib_axis) in sort(collect(axis_time_contrib); lt=(a, b)->(a[1] < b[1]))
            isempty(time_contrib_axis) && continue

            axis_total_time = mapreduce(x->x[2], +, collect(time_contrib_axis))
            axis_time[axis] = axis_total_time

            println("\nTime for each step of the $axis:          ( axis%) (total%)")
            for (step_label, step_time) in sort(collect(time_contrib_axis))
                @printf(" - %-25s %10.5f ms (%5.2f%%) (%5.2f%%)\n", 
                    step_label, step_time / 1e6, step_time / axis_total_time * 100, 
                    step_time / total_time * 100)
            end
            @printf(" => %-24s %10.5f ms          (%5.2f%%)\n", "Axis total time:", 
                axis_total_time / 1e6, axis_total_time / total_time * 100)
        end

        # Print the total distribution of time
        println("\nTotal time repartition: ")
        for (step_label, step_time) in sorted_time_contrib
            @printf(" - %-25s %10.5f ms (%5.2f%%)\n",
                    step_label, step_time / 1e6, step_time / total_time * 100)
        end

        @printf("\nAsynchronicity efficiency: %.2f sec / %.2f sec = %.2f%% (effective time / total steps time)\n",
            total_time / 1e9, sync_total_time / 1e9, total_time / sync_total_time * 100)
    end

    if is_root && params.measure_hw_counters && !isempty(axis_hw_counters)
        sorted_counters = sort(collect(axis_hw_counters); lt=(a, b)->(a[1] < b[1]))
        sorted_counters = map((p)->(string(first(p)) => last(p)), sorted_counters)
        if params.silent < 3
            print_hardware_counters_table(stdout, params.hw_counters_options, sorted_counters)
        end
        if !isempty(params.hw_counters_output)
            open(params.hw_counters_output, "w") do file
                print_hardware_counters_table(file, params.hw_counters_options, sorted_counters; raw_print=true)
            end
        end
    end

    if params.return_data
        return data, dt, cycles, cells_per_sec, sorted_time_contrib, async_efficiency
    else
        return dt, cycles, cells_per_sec, sorted_time_contrib, async_efficiency
    end
end
