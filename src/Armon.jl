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

export ArmonParameters, armon

# TODO LIST
# center the positions of the cells in the output file
# Remove most generics : 'where {T, V <: AbstractVector{T}}' etc... when T and V are not used in the method. Omitting the 'where' will not change anything.
# Bug: `conservation_vars` doesn't give correct values with MPI, even though the solution is correct
# Bug: steps are not properly categorized and filtered at the output, giving wrong asynchronicity efficiency
# Bug: some time measurements are incorrect on GPU
# Result struct/dict which holds all measured values (+ data if needed)
# Neighbour enum + `has_neighbour(params, side)` method

"""
    Axis

Enumeration of the axes of the domain
"""
@enum Axis X_axis Y_axis

# ROCKernels uses AMDGPU's ROCDevice, unlike CUDAKernels and KernelsAbstractions
GPUDevice = Union{Device, ROCDevice}

#
# Performance tracking
#

include("vtune_lib.jl")
using .VTune

include("perf_utils.jl")

#
# Main components
#

include("limiters.jl")
include("tests.jl")
include("parameters.jl")
include("data.jl")
include("domain_ranges.jl")
include("generic_kernel.jl")
include("timing_macros.jl")
include("kernels.jl")
include("halo_exchange.jl")
include("solver.jl")

end
