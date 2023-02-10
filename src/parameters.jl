
COMM = MPI.COMM_WORLD

function set_world_comm(comm::MPI.Comm)
    # Allows to customize which processes will be part of the grid
    global COMM = comm
end


mutable struct ArmonParameters{Flt_T}
    # Test problem type, riemann solver and solver scheme
    test::TestCase
    riemann::Symbol
    scheme::Symbol
    riemann_limiter::Limiter

    # Domain parameters
    nghost::Int
    nx::Int
    ny::Int
    dx::Flt_T
    domain_size::NTuple{2, Flt_T}
    origin::NTuple{2, Flt_T}
    cfl::Flt_T
    Dt::Flt_T
    cst_dt::Bool
    dt_on_even_cycles::Bool
    axis_splitting::Symbol
    projection::Symbol

    # Indexing
    row_length::Int
    col_length::Int
    nbcell::Int
    ideb::Int
    ifin::Int
    index_start::Int
    idx_row::Int
    idx_col::Int
    current_axis::Axis
    s::Int  # Stride
    stencil_width::Int

    # Bounds
    maxtime::Flt_T
    maxcycle::Int

    # Output
    silent::Int
    output_dir::String
    output_file::String
    write_output::Bool
    write_ghosts::Bool
    write_slices::Bool
    output_precision::Int
    animation_step::Int
    measure_time::Bool
    measure_hw_counters::Bool
    hw_counters_options::String
    hw_counters_output::String
    return_data::Bool

    # Performance
    use_threading::Bool
    use_simd::Bool
    use_gpu::Bool
    device::GPUDevice
    block_size::Int

    # MPI
    use_MPI::Bool
    is_root::Bool
    rank::Int
    root_rank::Int
    proc_size::Int
    proc_dims::NTuple{2, Int}
    cart_comm::MPI.Comm
    cart_coords::NTuple{2, Int}  # Coordinates of this process in the cartesian grid
    neighbours::NamedTuple{(:top, :bottom, :left, :right), NTuple{4, Int}}  # Ranks of the neighbours of this process
    global_grid::NTuple{2, Int}  # Dimensions (nx, ny) of the global grid
    single_comm_per_axis_pass::Bool
    extra_ring_width::Int  # Number of cells to compute additionally when 'single_comm_per_axis_pass' is true
    reorder_grid::Bool
    comm_array_size::Int

    # Asynchronicity
    async_comms::Bool

    # Tests & Comparison
    compare::Bool
    is_ref::Bool
    comparison_tolerance::Float64
end


# Constructor for ArmonParameters
function ArmonParameters(;
        ieee_bits = 64,
        test = :Sod, riemann = :acoustic, scheme = :GAD, projection = :euler,
        riemann_limiter = :minmod,
        nghost = 2, nx = 10, ny = 10, stencil_width = nothing,
        domain_size = nothing, origin = nothing,
        cfl = 0., Dt = 0., cst_dt = false, dt_on_even_cycles = false,
        axis_splitting = :Sequential,
        maxtime = 0, maxcycle = 500_000,
        silent = 0, output_dir = ".", output_file = "output",
        write_output = false, write_ghosts = false, write_slices = false, output_precision = nothing,
        animation_step = 0,
        measure_time = true, measure_hw_counters = false,
        hw_counters_options = nothing, hw_counters_output = nothing,
        use_threading = true, use_simd = true,
        use_gpu = false, device = :CUDA, block_size = 1024,
        use_MPI = true, px = 1, py = 1,
        single_comm_per_axis_pass = false, reorder_grid = true,
        async_comms = false,
        compare = false, is_ref = false, comparison_tolerance = 1e-10,
        return_data = false
    )

    flt_type = (ieee_bits == 64) ? Float64 : Float32

    if isnothing(output_precision)
        output_precision = flt_type == Float64 ? 17 : 9  # Exact output by default
    end

    # Make sure that all floating point types are the same
    cfl = flt_type(cfl)
    Dt = flt_type(Dt)
    maxtime = flt_type(maxtime)

    domain_size = isnothing(domain_size) ? nothing : Tuple(flt_type.(domain_size))
    origin = isnothing(origin) ? nothing : Tuple(flt_type.(origin))

    if cst_dt && Dt == zero(flt_type)
        error("Dt == 0 with constant step enabled")
    end

    if measure_hw_counters
        use_gpu && error("Hardware counters are not supported on GPU")
        async_comms && error("Hardware counters in an asynchronous context are NYI")
        !measure_time && error("Hardware counters are only done when timings are measured as well")

        hw_counters_options = @something hw_counters_options default_perf_options()
        hw_counters_output = @something hw_counters_output ""
    else
        hw_counters_options = ""
        hw_counters_output = ""
    end

    min_nghost = 1
    min_nghost += (scheme != :Godunov)
    min_nghost += single_comm_per_axis_pass
    min_nghost += (projection == :euler_2nd)

    if nghost < min_nghost
        error("Not enough ghost cells for the scheme and/or projection, at least $min_nghost are needed.")
    end

    if (nx % px != 0) || (ny % py != 0)
        error("The dimensions of the global domain ($nx x $ny) are not divisible by the number of processors ($px x $py)")
    end

    if projection == :none
        error("Lagrangian mode unsupported")
    end

    if isnothing(stencil_width)
        stencil_width = min_nghost
    elseif stencil_width < min_nghost
        @warn "The detected minimum stencil width is $min_nghost, but $stencil_width was given. \
               The Boundary conditions might be false." maxlog=1
    elseif stencil_width > nghost
        error("The stencil width given ($stencil_width) cannot be bigger than the number of ghost cells ($nghost)")
    end

    if riemann_limiter isa Symbol
        riemann_limiter = limiter_from_name(riemann_limiter)
    elseif !(riemann_limiter isa Limiter)
        error("Expected a Limiter type or a symbol, got: $riemann_limiter")
    end

    if test isa Symbol
        test_type = test_from_name(test)
        test = nothing
    elseif test isa TestCase
        test_type = typeof(test)
    else
        error("Expected a TestCase type or a symbol, got: $test")
    end

    if single_comm_per_axis_pass
        error("single_comm_per_axis_pass=true is broken")
    end

    # MPI
    if use_MPI
        !MPI.Initialized() && error("'use_MPI=true' but MPI has not yet been initialized")

        rank = MPI.Comm_rank(COMM)
        proc_size = MPI.Comm_size(COMM)

        # Create a cartesian grid communicator of px × py processes. reorder=true can be very
        # important for performance since it will optimize the layout of the processes.
        C_COMM = MPI.Cart_create(COMM, [Int32(px), Int32(py)], [Int32(0), Int32(0)], reorder_grid)
        (cx, cy) = MPI.Cart_coords(C_COMM)

        neighbours = (
            top    = MPI.Cart_shift(C_COMM, 1,  1)[2],
            bottom = MPI.Cart_shift(C_COMM, 1, -1)[2],
            left   = MPI.Cart_shift(C_COMM, 0, -1)[2],
            right  = MPI.Cart_shift(C_COMM, 0,  1)[2]
        )
    else
        rank = 0
        proc_size = 1
        C_COMM = COMM
        (cx, cy) = (0, 0)
        neighbours = (
            top    = MPI.PROC_NULL,
            bottom = MPI.PROC_NULL,
            left   = MPI.PROC_NULL,
            right  = MPI.PROC_NULL
        )
    end

    root_rank = 0
    is_root = rank == root_rank

    # GPU
    if use_gpu
        if device == :CUDA
            CUDA.allowscalar(false)
            device = CUDADevice()
        elseif device == :ROCM
            AMDGPU.allowscalar(false)
            device = ROCDevice()
        elseif device == :CPU
            is_root && @warn "`use_gpu=true` but the device is set to the CPU. Therefore no kernel will run on a GPU." maxlog=1
            device = CPU()  # Useful in some cases for debugging
        else
            error("Unknown GPU device: $device")
        end
    else
        device = CPU()
    end

    # Initialize the test
    if isnothing(domain_size)
        domain_size = default_domain_size(test_type)
        domain_size = Tuple(flt_type.(domain_size))
    end

    if isnothing(origin)
        origin = default_domain_origin(test_type)
        origin = Tuple(flt_type.(origin))
    end

    if isnothing(test)
        (sx, sy) = domain_size
        Δx::flt_type = sx / nx
        Δy::flt_type = sy / ny
        test = create_test(Δx, Δy, test_type)
    end

    if cfl == 0
        cfl = default_CFL(test)
    end

    if maxtime == 0
        maxtime = default_max_time(test)
    end

    # Dimensions of the global domain
    g_nx = nx
    g_ny = ny

    dx = flt_type(domain_size[1] / g_nx)

    # Dimensions of an array of the sub-domain
    nx ÷= px
    ny ÷= py
    row_length = nghost * 2 + nx
    col_length = nghost * 2 + ny
    nbcell = row_length * col_length

    # First and last index of the real domain of an array
    ideb = row_length * nghost + nghost + 1
    ifin = row_length * (ny - 1 + nghost) + nghost + nx
    index_start = ideb - row_length - 1  # Used only by the `@i` macro

    # Used only for indexing with the `@i` macro
    idx_row = row_length
    idx_col = 1

    # Ring width
    if single_comm_per_axis_pass
        extra_ring_width = 1
        extra_ring_width += projection == :euler_2nd
    else
        extra_ring_width = 0
    end

    if use_MPI
        comm_array_size = max(nx, ny) * nghost * 7
    else
        comm_array_size = 0
    end

    return ArmonParameters{flt_type}(
        test, riemann, scheme, riemann_limiter,
        nghost, nx, ny, dx, domain_size, origin,
        cfl, Dt, cst_dt, dt_on_even_cycles,
        axis_splitting, projection,
        row_length, col_length, nbcell,
        ideb, ifin, index_start,
        idx_row, idx_col,
        X_axis, 1, stencil_width,
        maxtime, maxcycle,
        silent, output_dir, output_file,
        write_output, write_ghosts, write_slices, output_precision, animation_step,
        measure_time,
        measure_hw_counters, hw_counters_options, hw_counters_output,
        return_data,
        use_threading, use_simd, use_gpu, device, block_size,
        use_MPI, is_root, rank, root_rank,
        proc_size, (px, py), C_COMM, (cx, cy), neighbours, (g_nx, g_ny),
        single_comm_per_axis_pass, extra_ring_width, reorder_grid, comm_array_size,
        async_comms,
        compare, is_ref, comparison_tolerance
    )
end


function print_parameters(p::ArmonParameters{T}) where T
    println("Parameters:")
    print(" - multithreading: ", p.use_threading)
    if p.use_threading
        if use_std_lib_threads
            println(" (Julia standard threads: ", Threads.nthreads(), ")")
        else
            println(" (Julia threads: ", Threads.nthreads(), ")")
        end
    else
        println()
    end
    println(" - use_simd:   ", p.use_simd)
    print(" - use_gpu:    ", p.use_gpu)
    if p.use_gpu
        print(", ")
        if p.device == CPU()
            println("CPU")
        elseif p.device == CUDADevice()
            println("CUDA")
        elseif p.device == ROCDevice()
            println("ROCm")
        else
            println("<unknown device>")
        end
        println(" - block size: ", p.block_size)
    else
        println()
    end
    println(" - use_MPI:    ", p.use_MPI)
    println(" - ieee_bits:  ", sizeof(T) * 8)
    println()
    println(" - test:       ", p.test)
    print(" - riemann:    ", p.riemann)
    if p.scheme != :Godunov
        println(", ", p.riemann_limiter)
    else
        println()
    end
    println(" - scheme:     ", p.scheme)
    println(" - splitting:  ", p.axis_splitting)
    println(" - cfl:        ", p.cfl)
    println(" - Dt:         ", p.Dt, p.dt_on_even_cycles ? ", updated only for even cycles" : "")
    println(" - euler proj: ", p.projection)
    println(" - cst dt:     ", p.cst_dt)
    println(" - stencil width: ", p.stencil_width)
    println(" - maxtime:    ", p.maxtime)
    println(" - maxcycle:   ", p.maxcycle)
    println()
    println(" - domain:     ", p.nx, "×", p.ny, " (", p.nghost, " ghosts)")
    println(" - domain size: ", join(p.domain_size, " × "), ", origin: (", join(p.origin, ", "), ")")
    println(" - nbcell:     ", @sprintf("%g", p.nx * p.ny), " (", p.nbcell, " total)")
    println(" - global:     ", p.global_grid[1], "×", p.global_grid[2])
    println(" - proc grid:  ", p.proc_dims[1], "×", p.proc_dims[2], " ($(p.reorder_grid ? "" : "not ")reordered)")
    println(" - coords:     ", p.cart_coords[1], "×", p.cart_coords[2], " (rank: ", p.rank, "/", p.proc_size-1, ")")
    println(" - comms per axis: ", p.single_comm_per_axis_pass ? 1 : 2)
    println(" - asynchronous communications: ", p.async_comms)
    println(" - measure step times: ", p.measure_time)
    if p.measure_hw_counters
        println(" - hardware counters measured: ", p.hw_counters_options)
    end
    println()
    if p.write_output
        println(" - write output: ", p.write_output, " (precision: ", p.output_precision, " digits)")
        println(" - write ghosts: ", p.write_ghosts)
        println(" - output file: ", p.output_file)
        if p.compare
            println(" - compare: ", p.compare, p.is_ref ? ", as reference" : "")
            println(" - tolerance: ", p.comparison_tolerance)
        end
        println()
    end
end


# Default copy method
function Base.copy(p::ArmonParameters{T}) where T
    return ArmonParameters([getfield(p, k) for k in fieldnames(ArmonParameters{T})]...)
end


function get_device_array(params::ArmonParameters)
    if params.device == CUDADevice()
        return CuArray
    elseif params.device == ROCDevice()
        return ROCArray
    else  # params.device == CPU()
        return Array
    end
end


function update_axis_parameters(params::ArmonParameters{T}, axis::Axis) where T
    (; row_length, global_grid, domain_size) = params
    (g_nx, g_ny) = global_grid
    (sx, sy) = domain_size

    params.current_axis = axis

    if axis == X_axis
        params.s = 1
        params.dx = sx / g_nx
    else  # axis == Y_axis
        params.s = row_length
        params.dx = sy / g_ny
    end
end

#
# Axis splitting
#

function split_axes(params::ArmonParameters{T}, cycle::Int) where T
    axis_1, axis_2 = X_axis, Y_axis
    if iseven(cycle)
        axis_1, axis_2 = axis_2, axis_1
    end

    if params.axis_splitting == :Sequential
        return [
            (X_axis, T(1.0)),
            (Y_axis, T(1.0)),
        ]
    elseif params.axis_splitting == :SequentialSym
        return [
            (axis_1, T(1.0)),
            (axis_2, T(1.0)),
        ]
    elseif params.axis_splitting == :Strang
        return [
            (axis_1, T(0.5)),
            (axis_2, T(1.0)),
            (axis_1, T(0.5)),
        ]
    elseif params.axis_splitting == :X_only
        return [(X_axis, T(1.0))]
    elseif params.axis_splitting == :Y_only
        return [(Y_axis, T(1.0))]
    else
        error("Unknown axes splitting method: $(params.axis_splitting)")
    end
end
