
#include "armon.h"


ptrdiff_t ArmonParams::offset_nghost = 0;
ptrdiff_t ArmonParams::offset_nx = 0;
ptrdiff_t ArmonParams::offset_ny = 0;
ptrdiff_t ArmonParams::offset_dx = 0;
ptrdiff_t ArmonParams::offset_domain_size = 0;
ptrdiff_t ArmonParams::offset_origin = 0;
ptrdiff_t ArmonParams::offset_row_length = 0;
ptrdiff_t ArmonParams::offset_s = 0;
ptrdiff_t ArmonParams::offset_stencil_width = 0;
ptrdiff_t ArmonParams::offset_cart_coords = 0;
ptrdiff_t ArmonParams::offset_global_grid = 0;
ptrdiff_t ArmonParams::offset_debug_indexes = 0;


ARMON_EXPORT bool init_params_offsets(const char** names, const int64_t* offsets, int field_count)
{
    std::unordered_map<std::string_view, ptrdiff_t*> fields_map{
            { "nghost",        &ArmonParams::offset_nghost        },
            { "nx",            &ArmonParams::offset_nx            },
            { "ny",            &ArmonParams::offset_ny            },
            { "dx",            &ArmonParams::offset_dx            },
            { "domain_size",   &ArmonParams::offset_domain_size   },
            { "origin",        &ArmonParams::offset_origin        },
            { "row_length",    &ArmonParams::offset_row_length    },
            { "s",             &ArmonParams::offset_s             },
            { "stencil_width", &ArmonParams::offset_stencil_width },
            { "cart_coords",   &ArmonParams::offset_cart_coords   },
            { "global_grid",   &ArmonParams::offset_global_grid   },
            { "debug_indexes", &ArmonParams::offset_debug_indexes },
    };

    for (int i = 0; i < field_count && !fields_map.empty(); i++) {
        std::string_view field_name(names[i]);
        auto pos = fields_map.find(field_name);
        if (pos == fields_map.end()) continue;
        *pos->second = offsets[i];
        fields_map.erase(pos);
    }

    if (!fields_map.empty()) {
        std::cerr << "'init_params_offsets' is missing " << fields_map.size() << " fields from ArmonParameters:\n";
        for (const auto& [name, _] : fields_map) {
            std::cerr << " - " << name << "\n";
        }
        return true; // Let Julia gracefully handle the error
    }

    return false;
}


int (*limiter_type_to_int)(void*);
int (*test_case_to_int)(void*);
void (*get_init_test_params)(void*, flt_t*, int);
void (*raise_exception_handler)(const char* msg) __attribute__((noreturn));


ARMON_EXPORT void init_callbacks(
        int (*limiter_type_to_int_fptr)(void*),
        int (*test_case_to_int_fptr)(void*),
        void (*get_init_test_params_fptr)(void*, flt_t*, int),
        void (*raise_exception_handler_fptr)(const char* msg) __attribute__((noreturn)))
{
    limiter_type_to_int = limiter_type_to_int_fptr;
    test_case_to_int = test_case_to_int_fptr;
    get_init_test_params = get_init_test_params_fptr;
    raise_exception_handler = raise_exception_handler_fptr;
}


ARMON_EXPORT int data_type_size()
{
    return sizeof(flt_t);
}


#if defined(__GNUC__)
#include <cxxabi.h>

const char* current_exception_typename(const std::exception&)
{
    int status;
    return abi::__cxa_demangle(abi::__cxa_current_exception_type()->name(), nullptr, nullptr, &status);
}
#else
#include <typeinfo>

const char* current_exception_typename(const std::exception& exception)
{
    return typeid(exception).name();
}
#endif


__attribute__((noreturn)) void raise_exception(const std::exception& exception)
{
    const char* exception_type = current_exception_typename(exception);
    auto str = "(" + std::string(exception_type) + ") " + exception.what();
    raise_exception_handler(str.c_str());
}
