#ifndef ARMON_CPP_LIMITERS_H
#define ARMON_CPP_LIMITERS_H

#include "armon.h"


template<Limiter L>
KOKKOS_INLINE_FUNCTION flt_t limiter(flt_t)
{
    static_assert(L == Limiter::None, "Wrong limiter type");
    return flt_t(1);
}


template<>
KOKKOS_INLINE_FUNCTION flt_t limiter<Limiter::Minmod>(flt_t r)
{
    return std::max(flt_t(0), std::min(flt_t(1), r));
}


template<>
KOKKOS_INLINE_FUNCTION flt_t limiter<Limiter::Superbee>(flt_t r)
{
    return std::max(std::max(flt_t(0), std::min(flt_t(1), 2*r)), std::min(flt_t(2), r));
}

#endif //ARMON_CPP_LIMITERS_H
