#ifndef ARMON_CPP_PARALLEL_KERNELS_H
#define ARMON_CPP_PARALLEL_KERNELS_H

#include <Kokkos_Core.hpp>

#include "armon.h"
#include "indexing.h"

#ifdef USE_SIMD_KERNELS
using RangeTeamType = Kokkos::TeamPolicy<Kokkos::IndexType<Idx>>;
using Team_t = RangeTeamType::member_type;
#endif


template<typename ScaleInfo, typename Functor>
void parallel_kernel(const RangeType& range, [[maybe_unused]] const ScaleInfo& range_info, const Functor& functor)
{
#ifdef USE_SIMD_KERNELS
    int league_size = static_cast<int>(range.end() - range.begin() + 1);
    Kokkos::parallel_for(RangeTeamType(league_size, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Team_t& team) {
        const Idx team_i = static_cast<Idx>(team.league_rank() * team.team_size()) + static_cast<Idx>(range.begin());
        const auto team_vector_range = Kokkos::ThreadVectorRange(team, team_i, team_i + team.team_size());
        Kokkos::parallel_for(team_vector_range, functor);
    });
#else
    Kokkos::parallel_for(range, functor);
#endif  // USE_SIMD_KERNELS
}


template<typename ScaleInfo, typename Functor, typename Reducer>
void parallel_reduce_kernel(const RangeType& range, [[maybe_unused]] const ScaleInfo& range_info, const Functor& functor, const Reducer& return_value)
{
#ifdef USE_SIMD_KERNELS
    int league_size = static_cast<int>(range.end() - range.begin() + 1);
    Kokkos::parallel_reduce(RangeTeamType(league_size, Kokkos::AUTO),
    KOKKOS_LAMBDA(const Team_t& team, typename Reducer::value_type& result) {
        const Idx team_i = static_cast<Idx>(team.league_rank() * team.team_size()) + static_cast<Idx>(range.begin());
        typename Reducer::value_type team_result;
        return_value.init(team_result);
        const auto team_vector_range = Kokkos::ThreadVectorRange(team, team_i, team_i + team.team_size());
        Kokkos::parallel_reduce(team_vector_range, functor, Reducer(team_result));
        if (team.team_rank() == 0) {
            return_value.join(result, team_result);
        }
    }, return_value);
#else
    Kokkos::parallel_reduce(range, functor, Kokkos::Min<flt_t>(return_value));
#endif  // USE_SIMD_KERNELS
}

#endif //ARMON_CPP_PARALLEL_KERNELS_H
