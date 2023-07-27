
import Armon: ArmonData, ArmonDualData, read_data_from_file, write_sub_domain_file, inflate, conservation_vars
import Armon: Side, Left, Right, Top, Bottom, has_neighbour, neighbour_at, border_domain, ghost_domain, offset_to
import Armon: read_border_array!, copy_to_send_buffer!, copy_from_recv_buffer!, write_border_array!
import Armon: get_recv_comm_array, get_send_comm_array, send_buffer, recv_buffer

using MPI
using CUDA: device as device_of
using AMDGPU

MPI.Init()
MPI.Barrier(MPI.COMM_WORLD)


TEST_KOKKOS_MPI = parse(Bool, get(ENV, "TEST_KOKKOS_MPI", "false"))
if TEST_KOKKOS_MPI
    using Kokkos

    if is_root && !Kokkos.is_initialized()
        Kokkos.load_wrapper_lib()
    end

    MPI.Barrier(MPI.COMM_WORLD)

    if !Kokkos.is_initialized()
        !is_root && Kokkos.load_wrapper_lib(; no_compilation=true, no_git=true)
        Kokkos.set_omp_vars()
        Kokkos.initialize()
    end
end


function read_sub_domain_from_global_domain_file!(params::ArmonParameters, data::ArmonData, file::IO)
    (g_nx, g_ny) = params.global_grid
    (cx, cy) = params.cart_coords
    (; nx, ny, nghost) = params

    # Ranges of the global domain
    global_cols = 1:g_nx
    global_rows = 1:g_ny
    if params.write_ghosts
        global_cols = inflate(global_cols, nghost)
        global_rows = inflate(global_rows, nghost)
    end

    # Ranges of the sub-domain
    col_range = 1:ny
    row_range = 1:nx
    if params.write_ghosts
        offset = nghost
    else
        offset = 0
    end

    # Position of the origin and end of this sub-domain
    pos_x = cx * length(row_range) + 1 - offset
    pos_y = cy * length(col_range) + 1 - offset
    end_x = pos_x + length(row_range) - 1 + offset * 2
    end_y = pos_y + length(col_range) - 1 + offset * 2
    col_offset = cy * length(col_range)

    # Ranges of the sub-domain in the global domain
    sub_domain_rows = pos_y:end_y
    cols_before = first(global_cols):(pos_x-1)
    cols_after = (end_x+1):last(global_cols)

    if params.write_ghosts
        col_range = inflate(col_range, nghost)
        row_range = inflate(row_range, nghost)
        offset = nghost
    else
        offset = 0
    end

    skip_cells(range) = for _ in range
        skipchars(!=('\n'), file)
        skip(file, 1)  # Skip the '\n'
    end

    for iy in global_rows
        if iy in sub_domain_rows
            skip_cells(cols_before)

            # `col_range = iy:iy` since we can only read one row at a time
            # The offset then transforms `iy` to the local index of the row
            col_range = (iy:iy) .- col_offset
            read_data_from_file(params, data, col_range, row_range, file)

            skip_cells(cols_after)
        else
            # Skip the entire row, made of g_nx cells (+ ghosts if any)
            skip_cells(global_cols)
        end

        skip(file, 1)  # Skip the additional '\n' at the end of each row of cells
    end
end


function ref_data_for_sub_domain(params::ArmonParameters{T}) where T
    file_path = get_reference_data_file_name(params.test, T)
    ref_data = ArmonData(params)
    ref_dt::T = 0
    ref_cycles = 0

    open(file_path, "r") do ref_file
        ref_dt = parse(T, readuntil(ref_file, ','))
        ref_cycles = parse(Int, readuntil(ref_file, '\n'))
        read_sub_domain_from_global_domain_file!(params, ref_data, ref_file)
    end

    return ref_dt, ref_cycles, ref_data
end


function ref_params_for_sub_domain(test::Symbol, type::Type, px, py; overriden_options...)
    ref_options = Dict{Symbol, Any}(
        :use_MPI => true, :px => px, :py => py, :reorder_grid => true, :async_comms => false
    )
    merge!(ref_options, overriden_options)
    return get_reference_params(test, type; ref_options...)
end


function set_comm_for_grid(px, py)
    new_grid_size = px * py
    global_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    # Only the first `new_grid_size` ranks will be part of the new communicator
    in_grid = global_rank < new_grid_size
    color = in_grid ? 0 : MPI.API.MPI_UNDEFINED[]
    sub_comm = MPI.Comm_split(MPI.COMM_WORLD, color, global_rank)
    return sub_comm, in_grid
end


macro MPI_test(comm, expr, kws...)
    # Similar to @test in Test.jl
    skip = [kw.args[2] for kw in kws if kw.args[1] === :skip]
    kws = filter(kw -> kw.args[1] ∉ (:skip, :broken), kws)
    length(skip) > 1 && error("'skip' only allowed once")
    length(kws) > 1 && error("Cannot handle keywords other than 'skip'")
    skip = length(skip) > 0 ? first(skip) : false

    # Run `expr` only if !skip, reduce the test result, then only the root prints and calls @test
    return esc(quote
        let comm = $comm, skip::Bool = $skip, test_rank_ok::Int = skip ? false : $expr;
            test_result = skip ? 0 : MPI.Allreduce(test_rank_ok, MPI.PROD, comm)
            test_result = test_result > 0
            if !test_result && !skip
                # Print which ranks failed the test
                test_results = MPI.Gather(test_rank_ok, 0, comm)
                ranks = MPI.Gather(MPI.Comm_rank(comm), 0, comm)
                if is_root
                    test_results = Bool.(test_results)
                    if !any(test_results)
                        println("All ranks failed this test:")
                    else
                        failed_ranks = ranks[.!test_results]
                        println("$(length(failed_ranks)) ranks failed this test (ranks: $failed_ranks):")
                    end
                end
            end
            is_root && @test test_result skip=skip
        end
    end)
end


macro root_test(expr, kws...)
    return esc(quote
        if is_root
            @test($expr, $(kws...))
        end
    end)
end


function test_neighbour_coords(px, py, proc_in_grid, global_comm)
    !proc_in_grid && return

    ref_params = ref_params_for_sub_domain(:Sod, Float64, px, py; global_comm)
    coords = ref_params.cart_coords

    for (coord, sides) in ((coords[1], (Left, Right)), (coords[2], (Bottom, Top))), 
            side in (coord % 2 == 0 ? sides : reverse(sides))
        has_neighbour(ref_params, side) || continue
        neighbour_rank = neighbour_at(ref_params, side)
        neighbour_coords = zeros(Int, 2)
        MPI.Sendrecv!(collect(coords), neighbour_coords, ref_params.cart_comm;
                      dest=neighbour_rank, source=neighbour_rank)
        neighbour_coords = tuple(neighbour_coords...)

        expected_coords = coords .+ offset_to(side)
        @test expected_coords == neighbour_coords
        if expected_coords != neighbour_coords
            @debug "[$(ref_params.rank)] $neighbour_rank at $side: expected $expected_coords, got $neighbour_coords"
        end
    end
end

NX = 100
NY = 100
using Printf

function dump_neighbours(px, py, proc_in_grid, global_comm)
    !proc_in_grid && return

    ref_params = ref_params_for_sub_domain(:Sod, Float64, px, py; nx=NX, ny=NY, global_comm)

    coords = ref_params.cart_coords

    neighbour_coords = Dict{Side, Tuple{Int, Int}}()

    for (coord, sides) in ((coords[1], (Left, Right)), (coords[2], (Bottom, Top))), 
            side in (coord % 2 == 0 ? sides : reverse(sides))
        has_neighbour(ref_params, side) || continue
        neighbour_rank = neighbour_at(ref_params, side)
        n_coords = zeros(Int, 2)
        MPI.Sendrecv!(collect(coords), n_coords, ref_params.cart_comm;
                      dest=neighbour_rank, source=neighbour_rank)
        neighbour_coords[side] = tuple(n_coords...)
    end

    MPI.Barrier(ref_params.cart_comm)

    # Wait for the previous rank
    if ref_params.rank > 0
        MPI.Recv(Bool, ref_params.cart_comm; source=ref_params.rank-1)
    end

    println("[$(ref_params.rank)]: $(coords)")
    for side in instances(Side)
        if has_neighbour(ref_params, side)
            neighbour_rank = neighbour_at(ref_params, side)
            @printf(" - %6s: [%2d] = %6s (expected: %6s)", string(side), neighbour_rank, string(neighbour_coords[side]), string(coords .+ offset_to(side)))
        else
            @printf(" - %6s: ∅", string(side))
        end
        println()
    end

    # Notify the next rank
    if ref_params.rank < px*py-1
        MPI.Send(true, ref_params.cart_comm; dest=ref_params.rank+1)
    end

    MPI.Barrier(ref_params.cart_comm)
end


function fill_domain_idx(array, domain, val)
    for (iy, j) in enumerate(domain.col), ix in 1:length(domain.row)
        i = ix + (j - 1)
        array[i] = val + iy * 1000 + ix
    end
end


function check_domain_idx(array, domain, val, my_rank, neighbour_rank)
    diff_count = 0
    for (iy, j) in enumerate(domain.col), ix in 1:length(domain.row)
        i = ix + j - 1
        expected_val = val + iy * 1000 + ix
        if expected_val != array[i]
            diff_count += 1
            @debug "[$my_rank] With $neighbour_rank: at ($i,$j) (or $ix,$iy), expected $expected_val, got $(Int(array[i]))"
        end
    end
    return diff_count
end


function test_halo_exchange(px, py, proc_in_grid, global_comm)
    !proc_in_grid && return

    ref_params = ref_params_for_sub_domain(:Sod, Int64, px, py; nx=NX, ny=NY, global_comm)
    data = ArmonDualData(ref_params)
    coords = ref_params.cart_coords

    for (coord, sides) in ((coords[1], (Left, Right)), (coords[2], (Bottom, Top))), 
            side in (coord % 2 == 0 ? sides : reverse(sides))
        has_neighbour(ref_params, side) || continue
        neighbour_rank = neighbour_at(ref_params, side)

        # Fill the domain we send with predictable data, with indexes encoded into the data
        domain = border_domain(ref_params, side)
        fill_domain_idx(device(data).rho, domain, ref_params.rank * 1_000_000)

        # "Halo exchange", but with one neighbour at a time
        comm_array = get_send_comm_array(data, side)
        read_border_array!(ref_params, data, comm_array, side)
        copy_to_send_buffer!(data, comm_array, side)
        wait(ref_params)

        requests = data.requests[side]
        MPI.Start(requests.send)
        MPI.Start(requests.recv)

        MPI.Wait(requests.send)
        MPI.Wait(requests.recv)

        comm_array = get_recv_comm_array(data, side)
        copy_from_recv_buffer!(data, comm_array, side)
        write_border_array!(ref_params, data, comm_array, side)
        wait(ref_params)

        # Check if the received array was correctly pasted into our ghost domain
        g_domain = ghost_domain(ref_params, side)
        diff_count = check_domain_idx(device(data).rho, g_domain, neighbour_rank * 1_000_000, ref_params.rank, neighbour_rank)

        @test diff_count == 0
    end
end


# All grid should be able to perfectly divide the number of cells in each direction in the reference
# case (100×100)
domain_combinations = [
    (1, 1),
    (1, 2),
    (1, 4),
    (4, 1),
    (2, 2),
    (4, 4),
    (5, 2),
    (2, 5),
    (5, 5)
]

total_proc_count = MPI.Comm_size(MPI.COMM_WORLD)

@testset "MPI" begin
    @testset "$(px)×$(py)" for (px, py) in domain_combinations
        enough_processes = px * py ≤ total_proc_count
        if enough_processes
            is_root && @info "Testing with a $(px)×$(py) domain"
            comm, proc_in_grid = set_comm_for_grid(px, py)
        else
            is_root && @info "Not enough processes to test a $(px)×$(py) domain"
            comm, proc_in_grid = MPI.COMM_NULL, false
        end

        # dump_neighbours(px, py, proc_in_grid, comm)

        @testset "Neighbours" begin
            test_neighbour_coords(px, py, proc_in_grid, comm)
        end

        @testset "Halo exchange" begin
            test_halo_exchange(px, py, proc_in_grid, comm)
        end

        @testset "Reference" begin
            @testset "$test with $type" for type in (Float64,),
                                            test in (:Sod, :Sod_y, :Sod_circ, :Sedov, :Bizarrium)
                @MPI_test comm begin
                    ref_params = ref_params_for_sub_domain(test, type, px, py; nx=NX, ny=NY, global_comm=comm)
                    dt, cycles, data = run_armon_reference(ref_params)
                    ref_dt, ref_cycles, ref_data = ref_data_for_sub_domain(ref_params)

                    @root_test dt ≈ ref_dt atol=abs_tol(type, ref_params.test) rtol=rel_tol(type, ref_params.test)
                    @root_test cycles == ref_cycles

                    diff_count, _ = count_differences(ref_params, host(data), ref_data)
                    if WRITE_FAILED
                        global_diff_count = MPI.Allreduce(diff_count, MPI.SUM, comm)
                        if global_diff_count > 0
                            write_sub_domain_file(ref_params, data, "test_$(test)_$(type)_$(px)x$(py)"; no_msg=true)
                            write_sub_domain_file(ref_params, ref_data, "ref_$(test)_$(type)_$(px)x$(py)"; no_msg=true)
                        end
                        println("[$(MPI.Comm_rank(comm))]: found $diff_count")
                    end

                    diff_count == 0
                end skip=!enough_processes || !proc_in_grid
            end
        end


        @testset "Async communications" begin
            @testset "$test with $type" for type in (Float64,),
                                            test in (:Sod, :Sod_y, :Sod_circ, :Sedov, :Bizarrium)
                @MPI_test comm begin
                    ref_params = ref_params_for_sub_domain(test, type, px, py; async_comms=true, nx=NX, ny=NY, global_comm=comm)
                    dt, cycles, data = run_armon_reference(ref_params)
                    ref_dt, ref_cycles, ref_data = ref_data_for_sub_domain(ref_params)

                    @root_test dt ≈ ref_dt atol=abs_tol(type, ref_params.test) rtol=rel_tol(type, ref_params.test)
                    @root_test cycles == ref_cycles

                    diff_count, _ = count_differences(ref_params, host(data), ref_data)
                    diff_count == 0
                end skip=!enough_processes || !proc_in_grid
            end
        end


        @testset "Conservation" begin
            @testset "$test" for test in (:Sod, :Sod_y, :Sod_circ)
                if enough_processes && proc_in_grid
                    ref_params = ref_params_for_sub_domain(test, Float64, px, py; maxcycle=10000, maxtime=10000, nx=NX, ny=NY, global_comm=comm)

                    data = ArmonDualData(ref_params)
                    init_test(ref_params, data)

                    init_mass, init_energy = conservation_vars(ref_params, data)
                    time_loop(ref_params, data)
                    end_mass, end_energy = conservation_vars(ref_params, data)
                else
                    init_mass, init_energy = 0., 0.
                    end_mass,  end_energy  = 0., 0.
                end

                @root_test   init_mass ≈ end_mass    atol=1e-12  skip=!enough_processes
                @root_test init_energy ≈ end_energy  atol=1e-12  skip=!enough_processes
            end
        end

        @testset "CUDA GPU" begin
            # no_cuda = !CUDA.has_cuda_gpu()
            # TODO: CUDA GPU MPI tests
        end

        @testset "ROCm GPU" begin
            # no_rocm = !AMDGPU.has_rocm_gpu()
            # TODO: CUDA GPU MPI tests
        end

        TEST_KOKKOS_MPI && @testset "Kokkos" begin
            @testset "$test with $type (async: $async_comms)" for type in (Float64,),
                                                                  test in (:Sod, :Sod_y, :Sod_circ, :Sedov, :Bizarrium),
                                                                  async_comms in (false, true)
                @MPI_test comm begin
                    ref_params = ref_params_for_sub_domain(test, type, px, py;
                        nx=NX, ny=NY, global_comm=comm,
                        use_kokkos=true, async_comms
                    )
                    dt, cycles, data = run_armon_reference(ref_params)
                    ref_dt, ref_cycles, ref_data = ref_data_for_sub_domain(ref_params)

                    @root_test dt ≈ ref_dt atol=abs_tol(type, ref_params.test) rtol=rel_tol(type, ref_params.test)
                    @root_test cycles == ref_cycles

                    diff_count, _ = count_differences(ref_params, host(data), ref_data)
                    if WRITE_FAILED
                        global_diff_count = MPI.Allreduce(diff_count, MPI.SUM, comm)
                        if global_diff_count > 0
                            write_sub_domain_file(ref_params, data, "test_$(test)_$(type)_$(px)x$(py)"; no_msg=true)
                            write_sub_domain_file(ref_params, ref_data, "ref_$(test)_$(type)_$(px)x$(py)"; no_msg=true)
                        end
                        println("[$(MPI.Comm_rank(comm))]: found $diff_count")
                    end

                    diff_count == 0
                end skip=!enough_processes || !proc_in_grid
            end
        end

        # TODO: thread pinning tests (no overlaps, no gaps)
        # TODO: GPU assignment tests (overlaps only if there is more processes than GPUs in a node)
        # TODO: add @debug statements for those tests to get a view of the structure of cores&gpus assigned to each rank

        MPI.free(comm)
    end
end
