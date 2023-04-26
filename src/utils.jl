
"""
    Axis

Enumeration of the axes of the domain
"""
@enum Axis X_axis Y_axis


"""
    Side

Enumeration of the sides of the domain
"""
@enum Side Left Right Bottom Top

sides_along(dir::Axis) = dir == X_axis ? (Left, Right) : (Bottom, Top)

function offset_to(side::Side)
    if side == Left
        return (-1, 0)
    elseif side == Right
        return (1, 0)
    elseif side == Bottom
        return (0, -1)
    elseif side == Top
        return (0, 1)
    end
end


# ROCKernels uses AMDGPU's ROCDevice, unlike CUDAKernels and KernelsAbstractions
GenericDevice = Union{Device, ROCDevice, Kokkos.ExecutionSpace}
