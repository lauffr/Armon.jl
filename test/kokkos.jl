
using Kokkos
if !Kokkos.is_initialized()
    Kokkos.set_omp_vars()
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
    return compare_with_reference_data(ref_params, dt, cycles, host(data), ref_data)
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
end
