cmake_minimum_required(VERSION 3.20)
project(armon_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(Kokkos REQUIRED)

option(USE_SINGLE_PRECISION "Use 'float' instead of 'double'" OFF)
option(CHECK_VIEW_ORDER "Check if all kernels are called properly" OFF)
option(TRY_ALL_CALLS "Surround all exported functions with a try-catch block" OFF)

add_library(armon_cpp SHARED
        armon.h armon.cpp
        indexing.h
        limiters.h
        utils.h
        kernels/acoustic.cpp
        kernels/perfect_gas_eos.cpp
        kernels/bizarrium_eos.cpp
        kernels/cell_update.cpp
        kernels/init_test.cpp
        kernels/euler_projection.cpp
        kernels/boundary_conditions.cpp
        kernels/reductions.cpp
        kernels/halo_exchange.cpp)

target_link_libraries(armon_cpp PUBLIC Kokkos::kokkos)
target_include_directories(armon_cpp PUBLIC ${CMAKE_SOURCE_DIR})
target_compile_options(armon_cpp PRIVATE
        $<$<CONFIG:Debug>:-Wall -Wextra -Wpedantic>)

if (USE_SINGLE_PRECISION)
    target_compile_definitions(armon_cpp PRIVATE USE_SINGLE_PRECISION)
endif()

if (CHECK_VIEW_ORDER)
    target_compile_definitions(armon_cpp PRIVATE CHECK_VIEW_ORDER)
endif()

if (TRY_ALL_CALLS OR (CMAKE_BUILD_TYPE STREQUAL "Debug"))
    target_compile_definitions(armon_cpp PRIVATE TRY_ALL_CALLS)
endif()
