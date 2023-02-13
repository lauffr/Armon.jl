
using Printf
import Armon: @i, @indexing_vars, ArmonData, init_test, time_loop, write_sub_domain_file, test_name


function cmp_cpu_with_reference(test::Symbol, type::Type; options...)
    ref_params = get_reference_params(test, type; options...)
    dt, cycles, data = run_armon_reference(ref_params)
    T = data_type(ref_params)
    ref_data = ArmonData(T, ref_params.nbcell, ref_params.comm_array_size)

    differences_count = compare_with_reference_data(ref_params, dt, cycles, data, ref_data)

    if differences_count > 0 && WRITE_FAILED
        file_name = "test_$(test_name(ref_params.test))_$(T)"
        ref_params.single_comm_per_axis_pass && (file_name *= "_single_comm")
        write_sub_domain_file(ref_params, data, file_name; no_msg=true)
    end

    return differences_count
end


function uninit_vars_propagation(test, type)
    ref_params = get_reference_params(test, type)

    data = ArmonData(data_type(ref_params), ref_params.nbcell, ref_params.comm_array_size)
    init_test(ref_params, data)

    for i in 1:ref_params.nbcell
        data.domain_mask == 0 || continue
        rho[i]  = 1e30
        Emat[i] = 1e30
        umat[i] = 1e30
        vmat[i] = 1e30
        pmat[i] = 1e30
        cmat[i] = 1e30
        ustar[i] = 1e30
        pstar[i] = 1e30
        work_array_1[i] = 1e30
        work_array_2[i] = 1e30
        work_array_3[i] = 1e30
        work_array_4[i] = 1e30
    end

    dt, cycles, _, _ = time_loop(ref_params, data, data)

    ref_data = ArmonData(data_type(ref_params), ref_params.nbcell, ref_params.comm_array_size)

    return compare_with_reference_data(ref_params, dt, cycles, data, ref_data)
end


@testset "Convergence" begin
    @testset "$test with $type" for type in (Float32, Float64),
                                    test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
        @test begin
            diff = cmp_cpu_with_reference(test, type)
            diff == 0
        end
    end

    @testset "Single boundary condition per pass" begin
        @testset "$test" for test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
            @test begin
                diff = cmp_cpu_with_reference(test, Float64; single_comm_per_axis_pass=true)
                diff == 0
            end skip=true  # TODO: all tests are broken since the indexing is still not 100% correct
        end
    end

    @testset "Uninitialized values propagation" begin
        @test begin
            diff = uninit_vars_propagation(:Sedov, Float64)
            diff == 0
        end
    end
end
