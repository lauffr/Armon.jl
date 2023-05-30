
using Kokkos
if !Kokkos.is_initialized()
    Kokkos.set_omp_vars()
    Kokkos.initialize()
end
Kokkos.require(; dims=[1], types=[Float64], exec_spaces=[Kokkos.OpenMP])


function run_armon_cpp_reference(ref_params::ArmonParameters)
    data = ArmonDualData(ref_params)
    init_test(ref_params, data)
    dt, cycles, _ = time_loop(ref_params, data)
    device_to_host!(data)
    return dt, cycles, data
end


function cmp_cpp_with_reference_for(type, test; kwargs...)
    cmake_options = ["-DCHECK_VIEW_ORDER=ON"]
    ref_params = get_reference_params(test, type; use_kokkos=true, cmake_options, kwargs...)
    dt, cycles, data = run_armon_cpp_reference(ref_params)
    ref_data = ArmonData(ref_params)
    return compare_with_reference_data(ref_params, dt, cycles, host(data), ref_data)
end


function cmp_halo_exchange_function(side; kwargs...)
    cmake_options = ["-DCHECK_VIEW_ORDER=ON"]
    kokkos_params = get_reference_params(:Sod, Float64; use_kokkos=true, debug_indexes=true, cmake_options, kwargs...)
    ref_params = get_reference_params(:Sod, Float64; use_kokkos=false, debug_indexes=true, kwargs...)

    kokkos_params.comm_array_size = max(kokkos_params.nx, kokkos_params.ny) * kokkos_params.nghost * 7
    ref_params.comm_array_size = max(ref_params.nx, ref_params.ny) * ref_params.nghost * 7

    kokkos_data = ArmonDualData(kokkos_params)
    kokkos_data_2 = ArmonDualData(kokkos_params)
    ref_data = ArmonDualData(ref_params)
    ref_data_2 = ArmonDualData(ref_params)

    # Set all elements to get rid of any uninitialized values (and potential NaNs)
    for f in fieldnames(typeof(device(ref_data)))
        kokkos_array = getproperty(device(kokkos_data), f)
        kokkos_array_2 = getproperty(device(kokkos_data_2), f)
        ref_array = getproperty(device(ref_data), f)
        ref_array_2 = getproperty(device(ref_data_2), f)
        kokkos_array .= 0
        kokkos_array_2 .= 0
        ref_array .= 0
        ref_array_2 .= 0
    end

    init_test(kokkos_params, kokkos_data)
    init_test(ref_params, ref_data)

    kokkos_comm_array = device(kokkos_data).work_array_1
    ref_comm_array = device(ref_data).work_array_1

    kokkos_comm_array = Kokkos.subview(kokkos_comm_array, Base.OneTo(kokkos_params.comm_array_size))
    ref_comm_array_v = view(ref_comm_array, Base.OneTo(ref_params.comm_array_size))

    @testset "Read" begin
        Armon.read_border_array!(kokkos_params, kokkos_data, kokkos_comm_array, side)
        Kokkos.fence()
    
        Armon.read_border_array!(ref_params, ref_data, ref_comm_array, side) |> wait
    
        @test kokkos_comm_array == ref_comm_array_v
    end

    @testset "Write" begin
        Armon.write_border_array!(kokkos_params, kokkos_data_2, kokkos_comm_array, side)
        Kokkos.fence()

        Armon.write_border_array!(ref_params, ref_data_2, ref_comm_array, side) |> wait

        @test device(kokkos_data_2).rho  == device(ref_data_2).rho
        @test device(kokkos_data_2).umat == device(ref_data_2).umat
        @test device(kokkos_data_2).vmat == device(ref_data_2).vmat
        @test device(kokkos_data_2).pmat == device(ref_data_2).pmat
        @test device(kokkos_data_2).cmat == device(ref_data_2).cmat
        @test device(kokkos_data_2).gmat == device(ref_data_2).gmat
        @test device(kokkos_data_2).Emat == device(ref_data_2).Emat
    end
end


@testset "Kokkos" begin
    @testset "$test with $type" for type in (Float64,),
                                    test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
        @test begin
            differences_count = cmp_cpp_with_reference_for(type, test)
            differences_count == 0
        end
    end

    @testset "Async $test" for test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
        @test begin
            differences_count = cmp_cpp_with_reference_for(Float64, test; async_comms=true)
            differences_count == 0
        end
    end

    @testset "Halo exchange" begin
        @testset "$side - $(nx)×$(ny)" for side in instances(Armon.Side),
                                           (nx, ny) in ((3, 10), (7, 10), (10, 7), (10, 10))
            cmp_halo_exchange_function(side; nx, ny)
        end
    end
end
