
#include "armon.h"


ARMON_EXPORT void boundaryConditions(
        void* p_ptr,
        int64_t loop_range_start, int64_t loop_range_step, int64_t loop_range_end,
        int64_t stride, int64_t i_start, int64_t disp,
        flt_t u_factor, flt_t v_factor,
        view& rho, view& umat, view& vmat, view& pmat, view& cmat, view& gmat, view& Emat)
ARMON_TRY {
    ArmonParams p{p_ptr};
    RangeType range_type{};
    RangeInfo1D range_info{};
    std::tie(range_type, range_info) = iter(loop_range_start, loop_range_step, loop_range_end);

    APPLY_6(CHECK_VIEW_LABEL, rho, umat, vmat, pmat, cmat, gmat);
    APPLY_1(CHECK_VIEW_LABEL, Emat);

    const Idx stencil_width = p.stencil_width();

    i_start += stride - 1;  // 0-index correction

    Kokkos::parallel_for(range_type,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx i = scale_index(lin_i, range_info);

        i = i * stride + i_start;
        Idx ip = i + disp;

        for (Idx w = 0; w < stencil_width; w++) {
            rho[i]  = rho[ip];
            umat[i] = umat[ip] * u_factor;
            vmat[i] = vmat[ip] * v_factor;
            pmat[i] = pmat[ip];
            cmat[i] = cmat[ip];
            gmat[i] = gmat[ip];
            Emat[i] = Emat[ip];

            i  -= disp;
            ip += disp;
        }
    });
} ARMON_CATCH
