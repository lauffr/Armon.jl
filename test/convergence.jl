
using Printf
import Armon: @i, @indexing_vars, ArmonData, init_test, time_loop, write_sub_domain_file, test_name
import Armon: Axis, X_axis, Y_axis, main_variables


function cmp_cpu_with_reference(test::Symbol, type::Type; options...)
    ref_params = get_reference_params(test, type; options...)
    dt, cycles, data = run_armon_reference(ref_params)
    T = data_type(ref_params)
    ref_data = ArmonData(ref_params)

    differences_count, max_diff = compare_with_reference_data(ref_params, dt, cycles, host(data), ref_data)

    if differences_count > 0 && WRITE_FAILED
        file_name = "test_$(test_name(ref_params.test))_$(T)"
        write_sub_domain_file(ref_params, data, file_name; no_msg=true)
    end

    @test differences_count == 0
    @test max_diff == 0
end


function axis_invariance(test::Symbol, type::Type, axis::Axis; options...)
    ref_params = get_reference_params(test, type; options...)
    dt, cycles, data = run_armon_reference(ref_params)

    (; nx, ny) = ref_params
    ng = ref_params.nghost
    lx = ref_params.row_length
    ly = ref_params.col_length

    if axis == X_axis
        r = (1:nx-1, :)
        r_offset = (2:nx, :)
    else
        r = (:, 1:ny-1)
        r_offset = (:, 2:ny)
    end

    vars = setdiff(main_variables(), (:x, :y))
    @testset "$var" for var in vars
        v_data = getfield(host(data), var)
        v_data = reshape(v_data, lx, ly)  # 1D to 2D array
        v_data = view(v_data, ng+1:lx-ng, ng+1:ly-ng)  # the nx × ny array of real (non-ghost) data

        # Substract each row/column with its neighbour => if the axis invariance is ok it should be 0
        # Note that we do not use `isapprox`, there is no tolerance here: invariance should be perfect.
        errors_count = count(!=(zero(type)), v_data[r...] .- v_data[r_offset...])

        @test errors_count == 0
    end
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

    differences_count, max_diff = compare_with_reference_data(ref_params, dt, cycles, host(data), ref_data)
    @test differences_count == 0
    @test max_diff == 0
end


@testset "Convergence" begin
    @testset "$test with $type" for type in (Float32, Float64),
                                    test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
        cmp_cpu_with_reference(test, type)
    end

    @testset "Axis invariance for $test" for (test, axis) in ([:Sod, Y_axis], [:Sod_y, X_axis], [:Bizarrium, Y_axis])
        axis_invariance(test, Float64, axis)
    end

    @testset "Uninitialized values propagation" begin
        uninit_vars_propagation(:Sedov, Float64)
    end

    @testset "Async code path" begin
        @testset "$test" for test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
            cmp_cpu_with_reference(test, Float64; async_comms=true, use_MPI=false)
        end
    end
end
