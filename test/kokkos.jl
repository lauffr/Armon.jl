
import Armon: write_sub_domain_file, test_name

using Kokkos
if !Kokkos.is_initialized()
    Kokkos.set_omp_vars()
    Kokkos.set_backends(split(TEST_KOKKOS_BACKEND, ',') .|> strip)
    Kokkos.initialize()
end


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
    
    differences_count, max_diff = compare_with_reference_data(ref_params, dt, cycles, host(data), ref_data)

    if differences_count > 0 && WRITE_FAILED
        file_name = "test_kokkos_$(test_name(ref_params.test))_$(data_type(ref_params))"
        write_sub_domain_file(ref_params, data, file_name; no_msg=true)
    end

    @test differences_count == 0
    @test max_diff == 0
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
    for f in fieldnames(typeof(host(ref_data)))
        kokkos_array = getproperty(host(kokkos_data), f)
        kokkos_array_2 = getproperty(host(kokkos_data_2), f)
        ref_array = getproperty(host(ref_data), f)
        ref_array_2 = getproperty(host(ref_data_2), f)
        kokkos_array .= 0
        kokkos_array_2 .= 0
        ref_array .= 0
        ref_array_2 .= 0

        copyto!(getproperty(device(kokkos_data), f), kokkos_array)
        copyto!(getproperty(device(kokkos_data_2), f), kokkos_array_2)
    end

    init_test(kokkos_params, kokkos_data)
    init_test(ref_params, ref_data)

    kokkos_comm_array = device(kokkos_data).work_array_1
    ref_comm_array = device(ref_data).work_array_1

    host_kokkos_comm_array = host(kokkos_data).work_array_1

    kokkos_comm_array = Kokkos.subview(kokkos_comm_array, Base.OneTo(kokkos_params.comm_array_size))
    host_kokkos_comm_array = Kokkos.subview(host_kokkos_comm_array, Base.OneTo(kokkos_params.comm_array_size))

    # `ref_comm_array` should be on the host too since the ref is on the host
    host_ref_comm_array = view(ref_comm_array, Base.OneTo(ref_params.comm_array_size))

    # data_1 to comm_array
    @testset "Read" begin
        Armon.read_border_array!(kokkos_params, kokkos_data, kokkos_comm_array, side)
        Kokkos.fence()

        Armon.read_border_array!(ref_params, ref_data, ref_comm_array, side) |> wait

        copyto!(host_kokkos_comm_array, kokkos_comm_array)
        @test host_kokkos_comm_array == host_ref_comm_array
    end

    # comm_array to data_2
    @testset "Write" begin
        Armon.write_border_array!(kokkos_params, kokkos_data_2, kokkos_comm_array, side)
        Kokkos.fence()

        Armon.write_border_array!(ref_params, ref_data_2, ref_comm_array, side) |> wait

        device_to_host!(kokkos_data_2)
        @test host(kokkos_data_2).rho  == host(ref_data_2).rho
        @test host(kokkos_data_2).umat == host(ref_data_2).umat
        @test host(kokkos_data_2).vmat == host(ref_data_2).vmat
        @test host(kokkos_data_2).pmat == host(ref_data_2).pmat
        @test host(kokkos_data_2).cmat == host(ref_data_2).cmat
        @test host(kokkos_data_2).gmat == host(ref_data_2).gmat
        @test host(kokkos_data_2).Emat == host(ref_data_2).Emat
    end
end


@testset "Kokkos" begin
    @testset "$test with $type" for type in (Float64,),
                                    test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
        cmp_cpp_with_reference_for(type, test)
    end

    @testset "Async $test" for test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
        cmp_cpp_with_reference_for(Float64, test; async_comms=true)
    end

    @testset "Halo exchange" begin
        @testset "$side - $(nx)×$(ny)" for side in instances(Armon.Side),
                                           (nx, ny) in ((3, 10), (7, 10), (10, 7), (10, 10))
            cmp_halo_exchange_function(side; nx, ny)
        end
    end
end
