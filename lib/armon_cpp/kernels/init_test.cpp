
#include "armon.h"

#include <variant>


struct Test_Sod {};
struct Test_Sod_y {};
struct Test_Sod_circ {};
struct Test_Bizarrium {};
struct Test_Sedov {
    flt_t r;
};


struct TestParams {
    flt_t high_rho;
    flt_t low_rho;
    flt_t high_E;
    flt_t low_E;
    flt_t high_u;
    flt_t low_u;
    flt_t high_v;
    flt_t low_v;
    flt_t test_var;
};


template<typename T>
KOKKOS_INLINE_FUNCTION bool test_region_high(flt_t, flt_t, T)
{
    static_assert(std::is_same_v<T, void>, "Invalid test case");
    return false;
}


template<>
KOKKOS_INLINE_FUNCTION bool test_region_high(flt_t x, flt_t, Test_Sod)
{
    return x <= 0.5;
}


template<>
KOKKOS_INLINE_FUNCTION bool test_region_high(flt_t, flt_t y, Test_Sod_y)
{
    return y <= 0.5;
}


template<>
KOKKOS_INLINE_FUNCTION bool test_region_high(flt_t x, flt_t y, Test_Sod_circ)
{
    return (Kokkos::pow(x - flt_t(0.5), flt_t(2)) + Kokkos::pow(y - flt_t(0.5), flt_t(2))) <= flt_t(0.125);
}


template<>
bool test_region_high(flt_t x, flt_t, Test_Bizarrium)
{
    return x <= 0.5;
}


template<>
bool test_region_high(flt_t x, flt_t y, Test_Sedov s)
{
    return Kokkos::pow(x, flt_t(2)) + Kokkos::pow(y, flt_t(2)) <= Kokkos::pow(s.r, flt_t(2));
}


template<typename TestCase>
void init_test(
        const ArmonParams& p,
        RangeType range_type,
        RangeInfo1D range_info,
        view& x, view& y, view& rho, view& Emat, view& umat, view& vmat,
        view& domain_mask, view& pmat, view& cmat, view& ustar, view& pstar,
        TestParams test_params, TestCase test, bool debug_indexes)
{
    const Idx nx = p.nx();
    const Idx ny = p.ny();
    const Idx row_length = p.row_length();
    const Idx nb_ghosts = p.nghost();

    const Idx cx = std::get<0>(p.cart_coords());
    const Idx cy = std::get<1>(p.cart_coords());

    const Idx pos_x = cx * nx;
    const Idx pos_y = cy * ny;

    const Idx g_nx = std::get<0>(p.global_grid());
    const Idx g_ny = std::get<1>(p.global_grid());

    const flt_t sx = std::get<0>(p.domain_size());
    const flt_t sy = std::get<1>(p.domain_size());

    const flt_t ox = std::get<0>(p.origin());
    const flt_t oy = std::get<1>(p.origin());

    Kokkos::parallel_for(range_type,
    KOKKOS_LAMBDA(const UIdx lin_i) {
        const Idx i = scale_index(lin_i, range_info);

        Idx ix = (i % row_length) - nb_ghosts;
        Idx iy = (i / row_length) - nb_ghosts;

        Idx g_ix = ix + pos_x;
        Idx g_iy = iy + pos_y;

        x[i] = flt_t(g_ix) / flt_t(g_nx) * sx + ox;
        y[i] = flt_t(g_iy) / flt_t(g_ny) * sy + oy;

        flt_t x_mid = x[i] + sx / flt_t(2 * g_nx);
        flt_t y_mid = y[i] + sy / flt_t(2 * g_ny);

        if (debug_indexes) {
            rho[i] = flt_t(i + 1);
            Emat[i] = flt_t(i + 1);
            umat[i] = flt_t(i + 1);
            vmat[i] = flt_t(i + 1);
        } else if (test_region_high(x_mid, y_mid, test)) {
            rho[i] = test_params.high_rho;
            Emat[i] = test_params.high_E;
            umat[i] = test_params.high_u;
            vmat[i] = test_params.high_v;
        }
        else {
            rho[i] = test_params.low_rho;
            Emat[i] = test_params.low_E;
            umat[i] = test_params.low_u;
            vmat[i] = test_params.low_v;
        }

        domain_mask[i] = (0 <= ix && ix < nx && 0 <= iy && iy < ny);

        // Set to zero to make sure no non-initialized values changes the result
        pmat[i] = 0;
        cmat[i] = 1;  // Set to 1 as a max speed of 0 will create NaNs
        ustar[i] = 0;
        pstar[i] = 0;
    });
}


ARMON_EXPORT void init_test(
        void* p_ptr,
        int64_t loop_range_start, int64_t loop_range_step, int64_t loop_range_end,
        view& x, view& y, view& rho, view& Emat, view& umat, view& vmat,
        view& domain_mask, view& pmat, view& cmat, view& ustar, view& pstar)
{
    ArmonParams p{p_ptr};

    APPLY_6(CHECK_VIEW_LABEL, x, y, rho, Emat, umat, vmat);
    APPLY_5(CHECK_VIEW_LABEL, domain_mask, pmat, cmat, ustar, pstar);

    TestParams test_params{};
    get_init_test_params(p.jl_value, (flt_t*) &test_params, sizeof(TestParams) / sizeof(flt_t));

    bool debug_indexes = p.debug_indexes();

    RangeType range_type{};
    RangeInfo1D range_info{};
    std::tie(range_type, range_info) = iter(
            loop_range_start, loop_range_step, loop_range_end
    );

    std::variant<Test_Sod, Test_Sod_y, Test_Sod_circ, Test_Bizarrium, Test_Sedov> test_case;
    switch (test_case_to_int(p.jl_value)) {
    default:
        std::cerr << "ERROR: unknown test case index: " << test_case_to_int(p.jl_value) << "\n";
        [[fallthrough]];
    case Sod:       test_case = Test_Sod{};                         break;
    case Sod_y:     test_case = Test_Sod_y{};                       break;
    case Sod_circ:  test_case = Test_Sod_circ{};                    break;
    case Bizarrium: test_case = Test_Bizarrium{};                   break;
    case Sedov:     test_case = Test_Sedov{test_params.test_var};  break;
    }

    std::visit([&](auto&& test) {
        init_test(
            p, range_type, range_info,
            x, y, rho, Emat, umat, vmat, domain_mask, pmat, cmat, ustar, pstar,
            test_params, test, debug_indexes
        );
    }, test_case);
}
