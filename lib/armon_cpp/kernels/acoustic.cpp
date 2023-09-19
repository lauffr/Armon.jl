
#include "armon.h"
#include "limiters.h"
#include "parallel_kernels.h"


KOKKOS_INLINE_FUNCTION std::tuple<flt_t, flt_t> acoustic_Godunov(
        flt_t rho_i, flt_t rho_im, flt_t c_i, flt_t c_im,
        flt_t u_i,   flt_t u_im,   flt_t p_i, flt_t p_im)
{
    flt_t rc_l = rho_im * c_im;
    flt_t rc_r = rho_i  * c_i;
    flt_t ustar_i = (rc_l * u_im + rc_r * u_i +               (p_im - p_i)) / (rc_l + rc_r);
    flt_t pstar_i = (rc_r * p_im + rc_l * p_i + rc_l * rc_r * (u_im - u_i)) / (rc_l + rc_r);
    return std::make_tuple(ustar_i, pstar_i);
}


ARMON_EXPORT void acoustic(
        void* p_ptr,
        int64_t main_range_start, int64_t main_range_step, int64_t main_range_end,
        int64_t row_range_start, int64_t row_range_step, int64_t row_range_end,
        view& ustar, view& pstar,
        const view& rho, const view& u, const view& pmat, const view& cmat)
ARMON_TRY {
    ArmonParams p{p_ptr};
    RangeType range_type{};
    RangeInfo range_info{};
    std::tie(range_type, range_info) = iter(
            main_range_start, main_range_step, main_range_end,
            row_range_start, row_range_step, row_range_end
    );

    APPLY_5(CHECK_VIEW_LABEL, ustar, pstar, rho, pmat, cmat);

    const Idx s = p.s();

    parallel_kernel(range_type, range_info,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        const Idx i = scale_index(lin_i, range_info);
        auto [ustar_i, pstar_i] = acoustic_Godunov(
            rho[i], rho[i-s], cmat[i], cmat[i-s],
              u[i],   u[i-s], pmat[i], pmat[i-s]
        );
        ustar[i] = ustar_i;
        pstar[i] = pstar_i;
    });
} ARMON_CATCH


template<Limiter L>
void acoustic_GAD(
        const ArmonParams& p,
        int64_t main_range_start, int64_t main_range_step, int64_t main_range_end,
        int64_t row_range_start, int64_t row_range_step, int64_t row_range_end,
        flt_t dt,
        view& ustar, view& pstar,
        const view& rho, const view& u, const view& pmat, const view& cmat)
{
    RangeType range_type{};
    RangeInfo range_info{};
    std::tie(range_type, range_info) = iter(
            main_range_start, main_range_step, main_range_end,
            row_range_start, row_range_step, row_range_end
    );

    const Idx s = p.s();
    const flt_t dx = p.dx();

    parallel_kernel(range_type, range_info,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        const Idx i = scale_index(lin_i, range_info);

        // First order acoustic solver on the left cell
        auto [ustar_im, pstar_im] = acoustic_Godunov(
            rho[i-s], rho[i-2*s], cmat[i-s], cmat[i-2*s],
              u[i-s],   u[i-2*s], pmat[i-s], pmat[i-2*s]
        );

        // First order acoustic solver on the current cell
        auto [ustar_i, pstar_i] = acoustic_Godunov(
            rho[i], rho[i-s], cmat[i], cmat[i-s],
              u[i],   u[i-s], pmat[i], pmat[i-s]
        );

        // First order acoustic solver on the right cell
        auto [ustar_ip, pstar_ip] = acoustic_Godunov(
            rho[i+s], rho[i], cmat[i+s], cmat[i],
              u[i+s],   u[i], pmat[i+s], pmat[i]
        );

        // Second order GAD acoustic solver on the current cell

        flt_t r_um = (ustar_ip -      u[i]) / (ustar_i -    u[i-s] + flt_t(1e-6));
        flt_t r_pm = (pstar_ip -   pmat[i]) / (pstar_i - pmat[i-s] + flt_t(1e-6));
        flt_t r_up = (   u[i-s] - ustar_im) / (   u[i] -   ustar_i + flt_t(1e-6));
        flt_t r_pp = (pmat[i-s] - pstar_im) / (pmat[i] -   pstar_i + flt_t(1e-6));

        r_um = limiter<L>(r_um);
        r_pm = limiter<L>(r_pm);
        r_up = limiter<L>(r_up);
        r_pp = limiter<L>(r_pp);

        flt_t dm_l = rho[i-s] * dx;
        flt_t dm_r = rho[i]   * dx;
        flt_t Dm   = (dm_l + dm_r) / 2;

        flt_t rc_l  = rho[i-s] * cmat[i-s];
        flt_t rc_r  = rho[i]   * cmat[i];
        flt_t theta = flt_t(0.5) * (1 - (rc_l + rc_r) / 2 * (dt / Dm));

        ustar[i] = ustar_i + theta * (r_up * (   u[i] - ustar_i) - r_um * (ustar_i -    u[i-s]));
        pstar[i] = pstar_i + theta * (r_pp * (pmat[i] - pstar_i) - r_pm * (pstar_i - pmat[i-s]));
    });
}


ARMON_EXPORT void acoustic_GAD(
        void* p_ptr,
        int64_t main_range_start, int64_t main_range_step, int64_t main_range_end,
        int64_t row_range_start, int64_t row_range_step, int64_t row_range_end,
        flt_t dt,
        view& ustar, view& pstar,
        const view& rho, const view& u, const view& pmat, const view& cmat)
ARMON_TRY {
    ArmonParams p{p_ptr};

    APPLY_3(CHECK_VIEW_LABEL, rho, pmat, cmat);

    switch (limiter_type_to_int(p.jl_value)) {
    case None:     return acoustic_GAD<None>(p, main_range_start, main_range_step, main_range_end, row_range_start, row_range_step, row_range_end, dt, ustar, pstar, rho, u, pmat, cmat);
    case Minmod:   return acoustic_GAD<Minmod>(p, main_range_start, main_range_step, main_range_end, row_range_start, row_range_step, row_range_end, dt, ustar, pstar, rho, u, pmat, cmat);
    case Superbee: return acoustic_GAD<Superbee>(p, main_range_start, main_range_step, main_range_end, row_range_start, row_range_step, row_range_end, dt, ustar, pstar, rho, u, pmat, cmat);
    }
} ARMON_CATCH
