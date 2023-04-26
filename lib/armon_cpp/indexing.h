#ifndef ARMON_CPP_INDEXING_H
#define ARMON_CPP_INDEXING_H

#include "Kokkos_Core.hpp"


using RangeType = Kokkos::RangePolicy<>;
using UIdx = RangeType::index_type;
using Idx = std::make_signed_t<UIdx>;


struct RangeInfo {
    Idx main_range_start;
    Idx main_range_step;
    Idx row_range_start;
    UIdx row_range_length;
};


struct RangeInfo1D {
    Idx start;
};


inline UIdx range_length(Idx start, Idx step, Idx stop)
{
    // See `length` in 'base/range.jl:762', simplified for the case where 'step >= 0'
    if ((start != stop) && ((step > 0) != (stop > start))) {
        return 0; // Empty range. See `isempty` in 'base/range.jl:668'
    }
    Idx diff = stop - start;
    return (diff / step) + 1;
}


inline std::tuple<RangeType, RangeInfo> iter(
        Idx main_range_start, Idx main_range_step, Idx main_range_end,
        Idx row_range_start, Idx row_range_step, Idx row_range_end)
{
    UIdx main_range_length = range_length(main_range_start, main_range_step, main_range_end);
    UIdx row_range_length = range_length(row_range_start, row_range_step, row_range_end);
    return {
        { 0, main_range_length * row_range_length },
        { main_range_start, main_range_step, row_range_start, row_range_length }
    };
}


inline std::tuple<RangeType, RangeInfo1D> iter(
        Idx loop_range_start, Idx loop_range_step, Idx loop_range_end)
{
    UIdx loop_range_length = range_length(loop_range_start, loop_range_step, loop_range_end);
    return {
            { 0, loop_range_length },
            { loop_range_start - 1 }
    };
}


/**
 * Transforms a linear 0-index to 2D indexes (into the 2D mesh) and back again into a linear 0-index.
 *
 * The input is an index into the range of number of cells to apply the kernel, while the output is a valid index into
 * the cell arrays.
 *
 * The transformation should be a mirror of the Julia version of the replacement of the `@index_2D_lin` macro (see
 * 'transform_kernel' in 'generic_kernels.jl').
 */
inline Idx scale_index(UIdx i, const RangeInfo& range_info)
{
    Idx ix = static_cast<Idx>(i / range_info.row_range_length);
    Idx iy = static_cast<Idx>(i % range_info.row_range_length);
    Idx j = (range_info.main_range_start + ix * range_info.main_range_step) - 1;
    return range_info.row_range_start + iy + j - /* to 0-index again */ 1;
}


inline Idx scale_index(UIdx i, const RangeInfo1D& range_info)
{
    return static_cast<Idx>(i) + range_info.start;
}

#endif //ARMON_CPP_INDEXING_H
