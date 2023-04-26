
#ifndef ARMON_CPP_UTILS_H
#define ARMON_CPP_UTILS_H


#define APPLY_IMPL(expr) expr

#define APPLY_1(f, arg1) \
    APPLY_IMPL(f(arg1))

#define APPLY_2(f, arg1, arg2) \
    APPLY_IMPL(f(arg1));       \
    APPLY_IMPL(f(arg2))

#define APPLY_3(f, arg1, arg2, arg3) \
    APPLY_IMPL(f(arg1));             \
    APPLY_IMPL(f(arg2));             \
    APPLY_IMPL(f(arg3))

#define APPLY_4(f, arg1, arg2, arg3, arg4) \
    APPLY_IMPL(f(arg1));                   \
    APPLY_IMPL(f(arg2));                   \
    APPLY_IMPL(f(arg3));                   \
    APPLY_IMPL(f(arg4))

#define APPLY_5(f, arg1, arg2, arg3, arg4, arg5) \
    APPLY_IMPL(f(arg1));                         \
    APPLY_IMPL(f(arg2));                         \
    APPLY_IMPL(f(arg3));                         \
    APPLY_IMPL(f(arg4));                         \
    APPLY_IMPL(f(arg5))

#define APPLY_6(f, arg1, arg2, arg3, arg4, arg5, arg6) \
    APPLY_IMPL(f(arg1));                               \
    APPLY_IMPL(f(arg2));                               \
    APPLY_IMPL(f(arg3));                               \
    APPLY_IMPL(f(arg4));                               \
    APPLY_IMPL(f(arg5));                               \
    APPLY_IMPL(f(arg6))


#ifdef CHECK_VIEW_ORDER
#define CHECK_VIEW_LABEL(v) if (v.label() != #v) std::cerr << "wrong view order in " << __PRETTY_FUNCTION__ << " : " << #v << "\n"
#else
#define CHECK_VIEW_LABEL(v) do { } while (false)
#endif


#endif //ARMON_CPP_UTILS_H
