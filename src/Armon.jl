module Armon

using Printf
using Polyester
using ThreadPinning
using KernelAbstractions
using MPI
using MacroTools
using AMDGPU
using ROCKernels
using CUDA
using CUDAKernels
using TimerOutputs

export ArmonParameters, ArmonDualData, SolverStats, armon, data_type, memory_required
export device_to_host!, host_to_device!, host, device, saved_variables, main_variables

# TODO LIST
# center the positions of the cells in the output file
# Remove most generics : 'where {T, V <: AbstractVector{T}}' etc... when T and V are not used in the method. Omitting the 'where' will not change anything.
# Rename some values in ArmonParameters & variables in ArmonData
# Make a custom generic reduction kernel applicable on a domain, which allows to remove `domain_mask`
# Monitor the removal of KA.jl's event system, and update the code accordingly

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

end
