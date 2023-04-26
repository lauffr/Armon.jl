#ifndef ARMON_CPP_ARMON_H
#define ARMON_CPP_ARMON_H

#include "Kokkos_Core.hpp"

#include "indexing.h"
#include "utils.h"


#ifdef USE_SINGLE_PRECISION
using flt_t = float;
#else
using flt_t = double;
#endif // USE_SINGLE_PRECISION


using view = Kokkos::View<flt_t*>;


enum Axis {
    X, Y
};

enum Limiter {
    None = 0,
    Minmod = 1,
    Superbee = 2
};

enum TestCase {
    Sod = 0,
    Sod_y = 1,
    Sod_circ = 2,
    Bizarrium = 3,
    Sedov = 4
};


struct ArmonParams {
    static ptrdiff_t offset_nghost;
    static ptrdiff_t offset_nx;
    static ptrdiff_t offset_ny;
    static ptrdiff_t offset_dx;
    static ptrdiff_t offset_domain_size;
    static ptrdiff_t offset_origin;
    static ptrdiff_t offset_row_length;
    static ptrdiff_t offset_s;
    static ptrdiff_t offset_stencil_width;

    void* jl_value;

    [[nodiscard]] char* ptr() const { return (char*) jl_value; }

    [[nodiscard]] int64_t nghost()            const { return *reinterpret_cast<int64_t*>(ptr() + offset_nghost); }
    [[nodiscard]] int64_t nx()                const { return *reinterpret_cast<int64_t*>(ptr() + offset_nx);     }
    [[nodiscard]] int64_t ny()                const { return *reinterpret_cast<int64_t*>(ptr() + offset_ny);     }
    [[nodiscard]] flt_t   dx()                const { return *reinterpret_cast<flt_t*  >(ptr() + offset_dx);     }
    [[nodiscard]] std::tuple<flt_t, flt_t> domain_size() const { return *reinterpret_cast<std::tuple<flt_t, flt_t>*>(ptr() + offset_domain_size); }
    [[nodiscard]] std::tuple<flt_t, flt_t> origin()      const { return *reinterpret_cast<std::tuple<flt_t, flt_t>*>(ptr() + offset_origin); }

    [[nodiscard]] int64_t row_length()    const { return *reinterpret_cast<int64_t*>(ptr() + offset_row_length);    }
    [[nodiscard]] Axis    s()             const { return *reinterpret_cast<Axis*   >(ptr() + offset_s);             }
    [[nodiscard]] int64_t stencil_width() const { return *reinterpret_cast<int64_t*>(ptr() + offset_stencil_width); }
};


#define ARMON_EXPORT extern "C"


extern int (*limiter_type_to_int)(void*);
extern int (*test_case_to_int)(void*);
extern void (*get_init_test_params)(void*, flt_t*, int);


#endif //ARMON_CPP_ARMON_H
