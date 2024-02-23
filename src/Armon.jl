module Armon

using Printf
using Polyester
using ThreadPinning
using KernelAbstractions
using MPI
using MacroTools
using TimerOutputs
using Preferences

export ArmonParameters, BlockGrid, SolverStats, armon, data_type, memory_required
export device_to_host!, host_to_device!

include("utils.jl")
include("domain_ranges.jl")
include("limiters.jl")
include("tests.jl")
include("parameters.jl")
include("data.jl")
include("blocking/blocking.jl")
include("profiling.jl")
include("generic_kernel.jl")
include("kernels.jl")
include("halo_exchange.jl")
include("io.jl")
include("solver.jl")

end
