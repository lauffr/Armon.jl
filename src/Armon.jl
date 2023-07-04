module Armon

using Printf
using Polyester
using ThreadPinning
using KernelAbstractions
using MPI
using MacroTools
using TimerOutputs

if !isdefined(Base, :get_extension)
    using Requires
end

export ArmonParameters, ArmonDualData, SolverStats, armon, data_type, memory_required
export device_to_host!, host_to_device!, host, device, saved_variables, main_variables

include("utils.jl")
include("domain_ranges.jl")
include("limiters.jl")
include("tests.jl")
include("parameters.jl")
include("data.jl")
include("generic_kernel.jl")
include("kernels.jl")
include("halo_exchange.jl")
include("io.jl")
include("solver.jl")

function __init__()
    @static if !isdefined(Base, :get_extension)
        @require AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e" include("../ext/ArmonAMDGPU.jl")
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("../ext/ArmonCUDA.jl")
        @require Kokkos = "3296cea9-b0de-4b57-aba0-ce554b517c3b" include("../ext/ArmonKokkos.jl")
    end
end

end
