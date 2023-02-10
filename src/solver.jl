
function time_loop(params::ArmonParameters{T}, data::ArmonData{V},
        cpu_data::ArmonData{W}) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; maxtime, maxcycle, nx, ny, silent, animation_step, is_root, dt_on_even_cycles) = params

    cycle  = 0
    t::T   = 0.
    next_dt::T = 0.
    prev_dt::T = 0.
    total_cycles_time::T = 0.

    t1 = time_ns()
    t_warmup = t1

    if params.use_MPI && params.use_gpu
        # Host version of temporary array used for MPI communications
        host_array = Vector{T}(undef, params.comm_array_size)
    else
        host_array = Vector{T}()
    end

    if params.async_comms
        # Disable multi-threading when computing the outer domains, since Polyester cannot run
        # multiple loops at the same time.
        outer_params = copy(params)
        outer_params.use_threading = false
    else
        outer_params = params
    end

    if silent <= 1
        initial_mass, initial_energy = conservation_vars(params, data)
    end

    update_axis_parameters(params, first(split_axes(params, cycle))[1])
    steps = steps_ranges(params)

    prev_event = NoneEvent()

    # Finalize the initialisation by calling the EOS on the entire domain
    update_EOS!(params, data, steps, :full) |> wait
    step_checkpoint(params, data, cpu_data, "update_EOS_init", cycle, params.current_axis) && @goto stop

    # Main solver loop
    while t < maxtime && cycle < maxcycle
        cycle_start = time_ns()

        if !dt_on_even_cycles || iseven(cycle)
            next_dt = dtCFL_MPI(params, data, prev_dt; dependencies=prev_event)
            prev_event = NoneEvent()

            if is_root && (!isfinite(next_dt) || next_dt <= 0.)
                error("Invalid dt for cycle $cycle: $next_dt")
            end

            if cycle == 0
                prev_dt = next_dt
            end
        end

        for (axis, dt_factor) in split_axes(params, cycle)
            update_axis_parameters(params, axis)
            steps = steps_ranges(params)

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

            @perf_task "loop" "EOS+comms+fluxes" @time_expr_c "EOS+comms+fluxes" if params.async_comms
                @sync begin
                    @async begin
                        event_2 = update_EOS!(params, data, steps, :inner; dependencies=prev_event)
                        event_2 = numericalFluxes!(params, data, prev_dt * dt_factor, steps, :inner; dependencies=event_2)
                        wait(event_2)
                    end

                    @async begin
                        # Since the other async tack is the one who should be using all the threads,
                        # here we forcefully disable multi-threading.
                        no_threading = true

                        event_1 = update_EOS!(outer_params, data, steps, :outer_lb; dependencies=prev_event, no_threading)
                        event_1 = update_EOS!(outer_params, data, steps, :outer_rt; dependencies=event_1, no_threading)

                        event_1 = boundaryConditions!(outer_params, data, host_array, axis; 
                            dependencies=event_1, no_threading)

                        event_1 = numericalFluxes!(outer_params, data, prev_dt * dt_factor, 
                            steps, :outer_lb; dependencies=event_1, no_threading)
                        event_1 = numericalFluxes!(outer_params, data, prev_dt * dt_factor, 
                            steps, :outer_rt; dependencies=event_1, no_threading)
                        wait(event_1)
                    end
                end

                step_checkpoint(params, data, cpu_data, "EOS+comms+fluxes", cycle, axis) && @goto stop
                event = NoneEvent()
            else
                event = update_EOS!(params, data, steps, :full; 
                    dependencies=prev_event)
                step_checkpoint(params, data, cpu_data, "update_EOS", cycle, axis; 
                    dependencies=event) && @goto stop

                event = boundaryConditions!(params, data, host_array, axis; dependencies=event)
                step_checkpoint(params, data, cpu_data, "boundaryConditions", cycle, axis; 
                    dependencies=event) && @goto stop

                event = numericalFluxes!(params, data, prev_dt * dt_factor, steps, :full; dependencies=event)
                step_checkpoint(params, data, cpu_data, "numericalFluxes", cycle, axis;
                    dependencies=event) && @goto stop

                params.measure_time && wait(event)
            end

            @perf_task "loop" "cellUpdate" event = cellUpdate!(params, data, prev_dt * dt_factor, steps;
                dependencies=event)
            step_checkpoint(params, data, cpu_data, "cellUpdate", cycle, axis; 
                dependencies=event) && @goto stop

            @perf_task "loop" "euler_proj" event = projection_remap!(params, data, host_array, steps,
                prev_dt * dt_factor; dependencies=event)
            step_checkpoint(params, data, cpu_data, "projection_remap", cycle, axis;
                dependencies=event) && @goto stop

            prev_event = event
        end

        if !is_warming_up()
            total_cycles_time += time_ns() - cycle_start
        end

        cycle += 1

        if is_root
            if silent <= 1
                wait(prev_event)
                current_mass, current_energy = conservation_vars(params, data)
                ΔM = abs(initial_mass - current_mass)     / initial_mass   * 100
                ΔE = abs(initial_energy - current_energy) / initial_energy * 100
                @printf("Cycle %4d: dt = %.18f, t = %.18f, |ΔM| = %#8.6g%%, |ΔE| = %#8.6g%%\n",
                    cycle, prev_dt, t, ΔM, ΔE)
            end
        elseif silent <= 1
            wait(prev_event)
            conservation_vars(params, data)
        end

        t += prev_dt
        prev_dt = next_dt

        if cycle == 5
            wait(prev_event)
            t_warmup = time_ns()
            set_warmup(false)
        end

        if animation_step != 0 && (cycle - 1) % animation_step == 0
            wait(prev_event)
            frame_index = (cycle - 1) ÷ animation_step
            frame_file = joinpath("anim", params.output_file) * "_" * @sprintf("%03d", frame_index)
            write_sub_domain_file(params, data, frame_file)
        end
    end

    wait(prev_event)

    @label stop

    t2 = time_ns()

    nb_cells = nx * ny
    grind_time = (t2 - t_warmup) / ((cycle - 5) * nb_cells)

    if is_root
        if params.compare
            # ignore timing errors when comparing
        elseif cycle <= 5 && maxcycle > 5
            error("More than 5 cycles are needed to compute the grind time, got: $cycle")
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
            println("Cycles:      ", cycle)
            println("Last cycle:  ", @sprintf("%.18f", t), " sec, Δt=", @sprintf("%.18f", next_dt), " sec")
        end
    end

    return next_dt, cycle, convert(T, 1 / grind_time), total_cycles_time
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
