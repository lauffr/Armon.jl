
using CUDA
using AMDGPU
using CUDAKernels
using ROCKernels


function run_armon_gpu_reference(ref_params::ArmonParameters)
    data = ArmonDualData(ref_params)
    init_test(ref_params, data)
    dt, cycles, _ = time_loop(ref_params, data)
    device_to_host!(data)
    return dt, cycles, data
end


function cmp_gpu_with_reference_for(type, test, device; kwargs...)
    ref_params = get_reference_params(test, type; use_gpu=true, device, kwargs...)
    dt, cycles, data = run_armon_gpu_reference(ref_params)
    ref_data = ArmonData(ref_params)
    differences_count, max_diff = compare_with_reference_data(ref_params, dt, cycles, host(data), ref_data)
    @test differences_count == 0
    @test max_diff == 0
end


@testset "GPU" begin
    CUDA.has_cuda_gpu() && @testset "CUDA" begin
        @testset "Reference" begin
            @testset "$test with $type" for type in (Float32, Float64),
                                            test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
                cmp_gpu_with_reference_for(type, test, :CUDA)
            end
        end

        @testset "Async" begin
            @testset "$test" for test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
                cmp_gpu_with_reference_for(Float64, test, :CUDA; async_comms=true)
            end
        end
    end

    AMDGPU.has_rocm_gpu() && @testset "ROCm" begin
        @testset "Reference" begin
            @testset "$test with $type" for type in (Float32, Float64),
                                            test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
                cmp_gpu_with_reference_for(type, test, :ROCM)
            end
        end

        @testset "Async" begin
            @testset "$test" for test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
                cmp_gpu_with_reference_for(Float64, test, :ROCM; async_comms=true)
            end
        end
    end
end
