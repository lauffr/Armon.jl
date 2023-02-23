
using Printf
import Armon: @i, @indexing_vars, ArmonData, init_test, time_loop, write_sub_domain_file, test_name


function cmp_cpu_with_reference(test::Symbol, type::Type; options...)
    ref_params = get_reference_params(test, type; options...)
    dt, cycles, data = run_armon_reference(ref_params)
    T = data_type(ref_params)
    ref_data = ArmonData(ref_params)

    differences_count = compare_with_reference_data(ref_params, dt, cycles, host(data), ref_data)

    if differences_count > 0 && WRITE_FAILED
        file_name = "test_$(test_name(ref_params.test))_$(T)"
        write_sub_domain_file(ref_params, data, file_name; no_msg=true)
    end

    return differences_count
end


function uninit_vars_propagation(test, type)
    ref_params = get_reference_params(test, type)

    data = ArmonDualData(ref_params)
    init_test(ref_params, data)

    mask = host(data).domain_mask
    big_val = type == Float32 ? 1e30 : 1e100
    for i in 1:ref_params.nbcell
        mask == 0 || continue
        rho[i]  = big_val
        Emat[i] = big_val
        umat[i] = big_val
        vmat[i] = big_val
        pmat[i] = big_val
        cmat[i] = big_val
        ustar[i] = big_val
        pstar[i] = big_val
        work_array_1[i] = big_val
        work_array_2[i] = big_val
        work_array_3[i] = big_val
        work_array_4[i] = big_val
    end

    dt, cycles, _ = time_loop(ref_params, data)

    ref_data = ArmonData(ref_params)

    return compare_with_reference_data(ref_params, dt, cycles, host(data), ref_data)
end


@testset "Convergence" begin
    @testset "$test with $type" for type in (Float32, Float64),
                                    test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
        @test begin
            diff = cmp_cpu_with_reference(test, type)
            diff == 0
        end
    end

    @testset "Uninitialized values propagation" begin
        @test begin
            diff = uninit_vars_propagation(:Sedov, Float64)
            diff == 0
        end
    end

    @testset "Async code path" begin
        @testset "$test" for test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
            @test begin
                diff = cmp_cpu_with_reference(test, Float64; async_comms=true, use_MPI=false)
                diff == 0
            end
        end
    end
end
