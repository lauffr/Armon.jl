
using Printf
using Dates
using Armon
import .Armon: ArmonDualData, DomainRange

const TEST_CPU    = parse(Bool, get(ENV, "TEST_CPU", "true"))
const TEST_CUDA   = parse(Bool, get(ENV, "TEST_CUDA", "false"))
const TEST_ROCM   = parse(Bool, get(ENV, "TEST_ROCM", "false"))
const TEST_KOKKOS = parse(Bool, get(ENV, "TEST_KOKKOS", "false"))
const TEST_TYPE   = eval(Meta.parse(get(ENV, "TEST_TYPE", "Float64")))
const TEST_N      = parse.(Float64, split(get(ENV, "TEST_N", "1e5,1e6,1e7,1e8"), ',') .|> strip)
const SAVE_TESTS  = parse(Bool, get(ENV, "SAVE_TESTS", "false"))

USE_GPU = false
USE_KOKKOS = false

if TEST_CUDA
    using CUDA
    DEVICE = :CUDA
    USE_GPU = true
    println("Testing CUDA")
elseif TEST_ROCM
    using AMDGPU
    DEVICE = :ROCM
    USE_GPU = true
    println("Testing ROCm")
elseif TEST_KOKKOS
    using Kokkos
    Kokkos.initialize()
    DEVICE = :KOKKOS
    USE_KOKKOS = true
    println("Testing Kokkos")
elseif TEST_CPU
    using ThreadPinning
    pinthreads(:compact)
    DEVICE = :CPU
    println("Testing CPU")
else
    error("No test enabled")
end


const ranges = Dict(
    :acoustic           => (p) -> p.steps_ranges.fluxes,
    :acoustic_GAD       => (p) -> p.steps_ranges.fluxes,
    :EOS_perfect_gas    => (p) -> p.steps_ranges.EOS,
    :EOS_bizarrium      => (p) -> p.steps_ranges.EOS,
    :cell_update        => (p) -> p.steps_ranges.cell_update,
    :first_order_remap  => (p) -> p.steps_ranges.advection,
    :second_order_remap => (p) -> p.steps_ranges.advection,
    :projection         => (p) -> p.steps_ranges.projection,
    :dtCFL              => (p) -> p.ideb:p.ifin
)


const kernels = [
    :acoustic           => (p, d, r) -> Armon.acoustic!(p, d, r, d.ustar, d.pstar, d.umat),
    :acoustic_GAD       => (p, d, r) -> Armon.acoustic_GAD!(p, d, r, p.cycle_dt, d.umat, p.riemann_limiter),
    :EOS_perfect_gas    => (p, d, r) -> Armon.update_perfect_gas_EOS!(p, d, r, Armon.data_type(p)(7/5)),
    :EOS_bizarrium      => (p, d, r) -> Armon.update_bizarrium_EOS!(p, d, r),
    :cell_update        => (p, d, r) -> Armon.cell_update!(p, d, r, p.cycle_dt, d.umat),
    :first_order_remap  => (p, d, r) -> Armon.first_order_euler_remap!(p, d, r, p.cycle_dt, d.work_array_1, d.work_array_2, d.work_array_3, d.work_array_4),
    :second_order_remap => (p, d, r) -> Armon.second_order_euler_remap!(p, d, r, p.cycle_dt, d.work_array_1, d.work_array_2, d.work_array_3, d.work_array_4),
    :projection         => (p, d, r) -> Armon.euler_projection!(p, d, r, p.cycle_dt, d.work_array_1, d.work_array_2, d.work_array_3, d.work_array_4),
    :dtCFL              => (p, d, _) -> Armon.local_time_step(p, d, p.cycle_dt),
    # Armon.init_test
    # Armon.boundaryConditions!
    # Armon.boundaryConditions!
    # Armon.read_border_array!
    # Armon.write_border_array!
]


function measure_perf(params::ArmonParameters, data::ArmonDualData, kernel, range; repeats=10)
    device_data = Armon.device(data)
    time = @elapsed begin
        for _ in 1:repeats
            kernel(params, device_data, range)
        end
        wait(params)
    end
    mean_time = time / repeats
    mean_perf = params.nbcell / mean_time
    return (; mean_time, mean_perf)
end


function measure_all_kernels(params::ArmonParameters, data::ArmonDualData; kwargs...)
    res = Dict()
    for (name, kernel_λ) in kernels
        range = ranges[name](params)
        res[name] = measure_perf(params, data, kernel_λ, range; kwargs...)
    end
    return res
end


function check_memory(params::ArmonParameters)
    mem_info = Armon.memory_info(params)
    mem_req  = Armon.memory_required(params)
    (mem_req * 1.05 > mem_info.total) && return false
    if mem_req * 1.05 < mem_info.free
        GC.gc(true)
    end
    return true
end


function setup_params(type, n)
    n = Int(round(sqrt(n)))
    return ArmonParameters(;
        ieee_bits=sizeof(type)*8,
        test=:Sod_circ, riemann_limiter=:minmod,
        nghost=5, nx=n, ny=n,
        silent=5, write_output=false, measure_time=false,
        use_MPI=false, async_comms=false,
        use_gpu=USE_GPU, device=DEVICE, block_size=128,
        use_kokkos=USE_KOKKOS
    )
end


function setup_data(params::ArmonParameters)
    if !check_memory(params)
        println("Not enough memory for this test")
        return nothing
    end

    data = ArmonDualData(params)
    Armon.init_test(params, data)
    Armon.time_step(params, data)
    wait(params)

    Armon.update_axis_parameters(params, Armon.X_axis)
    Armon.update_steps_ranges(params)
    params.cycle_dt = params.curr_cycle_dt

    return data
end


function disp_result(io::IO, res::Dict)
    pad_name = mapreduce(length ∘ string, max, keys(res)) + 5
    padding = [pad_name, 12, 12]
    for (text, pad) in zip(["Kernel", "Time [s]", "Perf [c/s]"], padding)
        print(io, text, ' ' ^ (pad - length(text)))
    end
    println(io)

    println(io, '─' ^ sum(padding))

    for (kernel_name, _) in kernels
        !haskey(res, kernel_name) && continue
        mean_time, mean_perf = res[kernel_name]
        for (val, pad) in zip([kernel_name, mean_time, mean_perf], padding)
            if val isa Float64
                float_fmt = Printf.Format("%#-$(pad).2g")
                text = Printf.format(float_fmt, val)
            else
                text = string(val)
            end
            print(io, text, ' ' ^ (pad - length(text)))
        end
        println(io)
    end

    println(io)
end

disp_result(res::Dict) = disp_result(stdout, res)


function result_csv(io::IO, res::Dict)
    sep = ", "
    join(io, ["Kernel", "Time [s]", "Perf [c/s]"], sep)

    for (kernel_name, _) in kernels
        !haskey(res, kernel_name) && continue
        println(io)
        mean_time, mean_perf = res[kernel_name]
        for val in [kernel_name, mean_time, mean_perf]
            if val isa Float64
                float_fmt = Printf.Format("%#.2g")
                text = Printf.format(float_fmt, val)
            else
                text = replace(string(val), "_" => "\\\\_")
            end
            print(io, text, ", ")
        end
        seek(io, position(io) - length(sep))
    end
end


function run_measurements(type, Ns; kwargs...)
    warmup_needed = true

    if SAVE_TESTS
        mkpath("./kernels_perfs")
        file_name = "results_$(DEVICE)_" * Dates.format(Dates.now(), DateFormat("yyyy-mm-dd-HH\\h-MM\\m"))
    end

    for n in Ns
        println("Measuring kernel perf for $n...")
        params = setup_params(type, n)
        data = setup_data(params)
        isnothing(data) && continue

        if warmup_needed
            println(params)
            SAVE_TESTS && open("./kernels_perfs/$(file_name)_INFO.txt", "w") do file
                println(file, params)
            end
            measure_all_kernels(params, data; kwargs...)
            warmup_needed = false
        end

        res = measure_all_kernels(params, data; kwargs...)
        disp_result(res)

        SAVE_TESTS && open("./kernels_perfs/$(file_name)_$(n).txt", "w") do file
            result_csv(file, res)
        end
    end
end


if !isinteractive()
    run_measurements(TEST_TYPE, TEST_N)
end
