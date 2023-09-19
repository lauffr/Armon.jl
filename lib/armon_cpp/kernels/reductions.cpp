
#include "armon.h"
#include "parallel_kernels.h"


ARMON_EXPORT flt_t dt_CFL(
        void* p_ptr,
        int64_t loop_range_start, int64_t loop_range_step, int64_t loop_range_end,
        const view& umat, const view& vmat, const view& cmat, const view& domain_mask,
        flt_t dx, flt_t dy)
ARMON_TRY {
    ArmonParams p{p_ptr};
    RangeType range_type{};
    RangeInfo1D range_info{};
    std::tie(range_type, range_info) = iter(
            loop_range_start, loop_range_step, loop_range_end
    );

    APPLY_4(CHECK_VIEW_LABEL, umat, vmat, cmat, domain_mask);

    flt_t dt = INFINITY;

    parallel_reduce_kernel(range_type, range_info,
    KOKKOS_LAMBDA(const UIdx lin_i, flt_t& dt_loop) {
        const Idx i = scale_index(lin_i, range_info);
        flt_t max_cx = Kokkos::max(Kokkos::abs(umat[i] + cmat[i]), Kokkos::abs(umat[i] - cmat[i])) * domain_mask[i];
        flt_t max_cy = Kokkos::max(Kokkos::abs(vmat[i] + cmat[i]), Kokkos::abs(vmat[i] - cmat[i])) * domain_mask[i];
        dt_loop = Kokkos::min(dt_loop, Kokkos::min(dx / max_cx, dy / max_cy));
    }, Kokkos::Min<flt_t>(dt));

    return dt;
} ARMON_CATCH


ARMON_EXPORT void conservation_vars(
        void* p_ptr,
        int64_t loop_range_start, int64_t loop_range_step, int64_t loop_range_end,
        const view& rho, const view& Emat, const view& domain_mask,
        flt_t* total_mass_p, flt_t* total_energy_p)
ARMON_TRY {
    ArmonParams p{p_ptr};
    RangeType range_type{};
    RangeInfo1D range_info{};
    std::tie(range_type, range_info) = iter(
            loop_range_start, loop_range_step, loop_range_end
    );

    APPLY_3(CHECK_VIEW_LABEL, rho, Emat, domain_mask);

    flt_t total_mass = 0;
    flt_t total_energy = 0;
    flt_t ds = p.dx() * p.dx();

    Kokkos::parallel_reduce(range_type,
    KOKKOS_LAMBDA(const UIdx lin_i, flt_t& mass, flt_t& energy) {
        const Idx i = scale_index(lin_i, range_info);
        flt_t cell_mass = rho[i] * domain_mask[i] * ds;
        flt_t cell_energy = cell_mass * Emat[i];
        mass += cell_mass;
        energy += cell_energy;
    }, Kokkos::Sum<flt_t>(total_mass), Kokkos::Sum<flt_t>(total_energy));

    *total_mass_p = total_mass;
    *total_energy_p = total_energy;
} ARMON_CATCH
