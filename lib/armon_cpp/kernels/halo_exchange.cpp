

#include "armon.h"


ARMON_EXPORT void read_border_array(
        void* p_ptr,
        int64_t main_range_start, int64_t main_range_step, int64_t main_range_end,
        int64_t row_range_start, int64_t row_range_step, int64_t row_range_end,
        int64_t side_length,
        const view& rho, const view& umat, const view& vmat, const view& pmat,
        const view& cmat, const view& gmat, const view& Emat,
        view& value_array)
ARMON_TRY {
    ArmonParams p{p_ptr};
    RangeType range_type{};
    RangeInfo range_info{};
    std::tie(range_type, range_info) = iter(
            main_range_start, main_range_step, main_range_end,
            row_range_start, row_range_step, row_range_end
    );

    APPLY_4(CHECK_VIEW_LABEL, rho, umat, vmat, pmat);
    APPLY_3(CHECK_VIEW_LABEL, cmat, gmat, Emat);

    const Idx nghost = p.nghost();

    Kokkos::parallel_for(range_type,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        const Idx idx = scale_index(lin_i, range_info);
        const Idx itr = static_cast<Idx>(lin_i);

        const Idx i   = itr / nghost;
        const Idx i_g = itr % nghost;
        const Idx i_arr = (i_g * side_length + i) * 7;

        value_array[i_arr+0] =  rho[idx];
        value_array[i_arr+1] = umat[idx];
        value_array[i_arr+2] = vmat[idx];
        value_array[i_arr+3] = pmat[idx];
        value_array[i_arr+4] = cmat[idx];
        value_array[i_arr+5] = gmat[idx];
        value_array[i_arr+6] = Emat[idx];
    });
} ARMON_CATCH


ARMON_EXPORT void write_border_array(
        void* p_ptr,
        int64_t main_range_start, int64_t main_range_step, int64_t main_range_end,
        int64_t row_range_start, int64_t row_range_step, int64_t row_range_end,
        int64_t side_length,
        view& rho, view& umat, view& vmat, view& pmat,
        view& cmat, view& gmat, view& Emat,
        const view& value_array)
ARMON_TRY {
    ArmonParams p{p_ptr};
    RangeType range_type{};
    RangeInfo range_info{};
    std::tie(range_type, range_info) = iter(
            main_range_start, main_range_step, main_range_end,
            row_range_start, row_range_step, row_range_end
    );

    APPLY_4(CHECK_VIEW_LABEL, rho, umat, vmat, pmat);
    APPLY_3(CHECK_VIEW_LABEL, cmat, gmat, Emat);

    const Idx nghost = p.nghost();

    Kokkos::parallel_for(range_type,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        const Idx idx = scale_index(lin_i, range_info);
        const Idx itr = static_cast<Idx>(lin_i);

        const Idx i   = itr / nghost;
        const Idx i_g = itr % nghost;
        const Idx i_arr = (i_g * side_length + i) * 7;

         rho[idx] = value_array[i_arr+0];
        umat[idx] = value_array[i_arr+1];
        vmat[idx] = value_array[i_arr+2];
        pmat[idx] = value_array[i_arr+3];
        cmat[idx] = value_array[i_arr+4];
        gmat[idx] = value_array[i_arr+5];
        Emat[idx] = value_array[i_arr+6];
    });
} ARMON_CATCH
