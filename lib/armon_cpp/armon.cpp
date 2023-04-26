
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


ARMON_EXPORT bool init_params_offsets(const char** names, const int64_t* offsets, int field_count)
{
    std::unordered_map<std::string_view, ptrdiff_t*> fields_map{
            { "nghost",            &ArmonParams::offset_nghost            },
            { "nx",                &ArmonParams::offset_nx                },
            { "ny",                &ArmonParams::offset_ny                },
            { "dx",                &ArmonParams::offset_dx                },
            { "domain_size",       &ArmonParams::offset_domain_size       },
            { "origin",            &ArmonParams::offset_origin            },
            { "row_length",        &ArmonParams::offset_row_length        },
            { "s",                 &ArmonParams::offset_s                 },
            { "stencil_width",     &ArmonParams::offset_stencil_width     },
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


ARMON_EXPORT void init_callbacks(
        int (*limiter_type_to_int_fptr)(void*),
        int (*test_case_to_int_fptr)(void*),
        void (*get_init_test_params_fptr)(void*, flt_t*, int))
{
    limiter_type_to_int = limiter_type_to_int_fptr;
    test_case_to_int = test_case_to_int_fptr;
    get_init_test_params = get_init_test_params_fptr;
}


ARMON_EXPORT int data_type_size()
{
    return sizeof(flt_t);
}
