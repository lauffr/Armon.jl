
#include "armon.h"


ARMON_EXPORT void cell_update(
        void* p_ptr,
        int64_t main_range_start, int64_t main_range_step, int64_t main_range_end,
        int64_t row_range_start, int64_t row_range_step, int64_t row_range_end,
        flt_t dt,
        const view& ustar, const view& pstar,
        view& rho, view& u, view& Emat)
ARMON_TRY {
    ArmonParams p{p_ptr};
    RangeType range_type{};
    RangeInfo range_info{};
    std::tie(range_type, range_info) = iter(
            main_range_start, main_range_step, main_range_end,
            row_range_start, row_range_step, row_range_end
    );

    APPLY_4(CHECK_VIEW_LABEL, ustar, pstar, rho, Emat);

    const Idx s = p.s();
    const flt_t dx = p.dx();

    Kokkos::parallel_for(range_type,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        const Idx i = scale_index(lin_i, range_info);
        flt_t dm = rho[i] * dx;
        rho[i]   = dm / (dx + dt * (ustar[i+s] - ustar[i]));
        u[i]    += dt / dm * (pstar[i]            - pstar[i+s]             );
        Emat[i] += dt / dm * (pstar[i] * ustar[i] - pstar[i+s] * ustar[i+s]);
    });
} ARMON_CATCH
