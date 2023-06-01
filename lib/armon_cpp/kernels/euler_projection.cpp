
#include "armon.h"


ARMON_EXPORT void euler_projection(
        void* p_ptr,
        int64_t main_range_start, int64_t main_range_step, int64_t main_range_end,
        int64_t row_range_start, int64_t row_range_step, int64_t row_range_end,
        flt_t dt,
        const view& ustar,
        view& rho, view& umat, view& vmat, view& Emat,
        view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho)
ARMON_TRY {
    ArmonParams p{p_ptr};
    RangeType range_type{};
    RangeInfo range_info{};
    std::tie(range_type, range_info) = iter(
            main_range_start, main_range_step, main_range_end,
            row_range_start, row_range_step, row_range_end
    );

    APPLY_5(CHECK_VIEW_LABEL, ustar, rho, umat, vmat, Emat);

    const Idx s = p.s();
    const flt_t dx = p.dx();

    Kokkos::parallel_for(range_type,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        const Idx i = scale_index(lin_i, range_info);

        flt_t dX = dx + dt * (ustar[i+s] - ustar[i]);

        flt_t tmp_rho  = (dX * rho[i]           - (advection_rho[i+s]  - advection_rho[i] )) / dx;
        flt_t tmp_urho = (dX * rho[i] * umat[i] - (advection_urho[i+s] - advection_urho[i])) / dx;
        flt_t tmp_vrho = (dX * rho[i] * vmat[i] - (advection_vrho[i+s] - advection_vrho[i])) / dx;
        flt_t tmp_Erho = (dX * rho[i] * Emat[i] - (advection_Erho[i+s] - advection_Erho[i])) / dx;

        rho[i]  = tmp_rho;
        umat[i] = tmp_urho / tmp_rho;
        vmat[i] = tmp_vrho / tmp_rho;
        Emat[i] = tmp_Erho / tmp_rho;
    });
} ARMON_CATCH


ARMON_EXPORT void first_order_euler_remap(
        void* p_ptr,
        int64_t main_range_start, int64_t main_range_step, int64_t main_range_end,
        int64_t row_range_start, int64_t row_range_step, int64_t row_range_end,
        flt_t dt,
        const view& ustar, const view& rho, const view& umat, const view& vmat, const view& Emat,
        view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho)
ARMON_TRY {
    ArmonParams p{p_ptr};
    RangeType range_type{};
    RangeInfo range_info{};
    std::tie(range_type, range_info) = iter(
            main_range_start, main_range_step, main_range_end,
            row_range_start, row_range_step, row_range_end
    );

    APPLY_5(CHECK_VIEW_LABEL, ustar, rho, umat, vmat, Emat);

    const Idx s = p.s();

    Kokkos::parallel_for(range_type,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx i = scale_index(lin_i, range_info);

        flt_t disp = dt * ustar[i];
        Idx is = i;
        i -= (disp > 0) * s;

        advection_rho[is]  = disp * (rho[i]          );
        advection_urho[is] = disp * (rho[i] * umat[i]);
        advection_vrho[is] = disp * (rho[i] * vmat[i]);
        advection_Erho[is] = disp * (rho[i] * Emat[i]);
    });
} ARMON_CATCH


KOKKOS_INLINE_FUNCTION flt_t slope_minmod(flt_t u_im, flt_t u_i, flt_t u_ip, flt_t r_m, flt_t r_p)
{
    flt_t D_u_p = r_p * (u_ip - u_i );
    flt_t D_u_m = r_m * (u_i  - u_im);
    flt_t s = Kokkos::copysign(flt_t(1), D_u_p);
    return s * Kokkos::max(flt_t(0), Kokkos::min(s * D_u_p, s * D_u_m));
}


ARMON_EXPORT void second_order_euler_remap(
        void* p_ptr,
        int64_t main_range_start, int64_t main_range_step, int64_t main_range_end,
        int64_t row_range_start, int64_t row_range_step, int64_t row_range_end,
        flt_t dt,
        const view& ustar, const view& rho, const view& umat, const view& vmat, const view& Emat,
        view& advection_rho, view& advection_urho, view& advection_vrho, view& advection_Erho)
ARMON_TRY {
    ArmonParams p{p_ptr};
    RangeType range_type{};
    RangeInfo range_info{};
    std::tie(range_type, range_info) = iter(
            main_range_start, main_range_step, main_range_end,
            row_range_start, row_range_step, row_range_end
    );

    APPLY_5(CHECK_VIEW_LABEL, ustar, rho, umat, vmat, Emat);

    const Idx s = p.s();
    const flt_t dx = p.dx();

    Kokkos::parallel_for(range_type,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        Idx i = scale_index(lin_i, range_info);

        Idx is = i;
        flt_t disp = dt * ustar[i];
        flt_t Dx;
        if (disp > 0) {
            Dx = -(dx - dt * ustar[i-s]);
            i = i - s;
        } else {
            Dx = dx + dt * ustar[i+s];
        }

        flt_t Dx_lm = dx + dt * (ustar[i]     - ustar[i-s]);
        flt_t Dx_l  = dx + dt * (ustar[i+s]   - ustar[i]  );
        flt_t Dx_lp = dx + dt * (ustar[i+2*s] - ustar[i+s]);

        flt_t r_m = (2 * Dx_l) / (Dx_l + Dx_lm);
        flt_t r_p = (2 * Dx_l) / (Dx_l + Dx_lp);

        flt_t slope_r  = slope_minmod(rho[i-s]            , rho[i]          , rho[i+s]            , r_m, r_p);
        flt_t slope_ur = slope_minmod(rho[i-s] * umat[i-s], rho[i] * umat[i], rho[i+s] * umat[i+s], r_m, r_p);
        flt_t slope_vr = slope_minmod(rho[i-s] * vmat[i-s], rho[i] * vmat[i], rho[i+s] * vmat[i+s], r_m, r_p);
        flt_t slope_Er = slope_minmod(rho[i-s] * Emat[i-s], rho[i] * Emat[i], rho[i+s] * Emat[i+s], r_m, r_p);

        flt_t length_factor = Dx / (2 * Dx_l);
        advection_rho[is]  = disp * (rho[i]           - slope_r  * length_factor);
        advection_urho[is] = disp * (rho[i] * umat[i] - slope_ur * length_factor);
        advection_vrho[is] = disp * (rho[i] * vmat[i] - slope_vr * length_factor);
        advection_Erho[is] = disp * (rho[i] * Emat[i] - slope_Er * length_factor);
    });
} ARMON_CATCH
