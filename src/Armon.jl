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

export ArmonParameters, ArmonDualData, armon, data_type, memory_required
export device_to_host!, host_to_device!, host, device, saved_variables, main_variables

# TODO LIST
# center the positions of the cells in the output file
# Remove most generics : 'where {T, V <: AbstractVector{T}}' etc... when T and V are not used in the method. Omitting the 'where' will not change anything.
# Bug: steps are not properly categorized and filtered at the output, giving wrong asynchronicity efficiency
# Bug: some time measurements are incorrect on GPU
# Result struct/dict which holds all measured values (+ data if needed)
# Rename some values in ArmonParameters & variables in ArmonData
# Make a custom generic reduction kernel applicable on a domain, which allows to remove `domain_mask`
# Monitor the removal of KA.jl's event system, and update the code accordingly

#
# Performance tracking
#

include("vtune_lib.jl")
using .VTune

include("perf_utils.jl")

#
# Main components
#

include("utils.jl")
include("domain_ranges.jl")
include("limiters.jl")
include("tests.jl")
include("parameters.jl")
include("data.jl")
include("generic_kernel.jl")
include("timing_macros.jl")
include("kernels.jl")
include("halo_exchange.jl")
include("io.jl")
include("solver.jl")

end
