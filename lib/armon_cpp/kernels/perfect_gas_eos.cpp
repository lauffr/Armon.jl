
#include "armon.h"
#include "parallel_kernels.h"


ARMON_EXPORT void update_perfect_gas_EOS(
        [[maybe_unused]] void* p_ptr,
        int64_t main_range_start, int64_t main_range_step, int64_t main_range_end,
        int64_t row_range_start, int64_t row_range_step, int64_t row_range_end,
        flt_t gamma,
        const view& rho, const view& Emat, const view& umat, const view& vmat,
        view& pmat, view& cmat, view& gmat)
ARMON_TRY {
    // ArmonParams p{p_ptr};
    RangeType range_type{};
    RangeInfo range_info{};
    std::tie(range_type, range_info) = iter(
            main_range_start, main_range_step, main_range_end,
            row_range_start, row_range_step, row_range_end
    );

    APPLY_6(CHECK_VIEW_LABEL, rho, umat, vmat, Emat, pmat, cmat);
    APPLY_1(CHECK_VIEW_LABEL, gmat);

    parallel_kernel(range_type, range_info,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        const Idx i = scale_index(lin_i, range_info);
        flt_t e = Emat[i] - flt_t(0.5) * (Kokkos::pow(umat[i], flt_t(2)) + Kokkos::pow(vmat[i], flt_t(2)));
        pmat[i] = (gamma - 1) * rho[i] * e;
        cmat[i] = Kokkos::sqrt(gamma * pmat[i] / rho[i]);
        gmat[i] = (1 + gamma) / 2;
    });
} ARMON_CATCH
