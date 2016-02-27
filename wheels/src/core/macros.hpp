#pragma once

#if !defined(NDEBUG)
#define wheels_debug
#else
#define wheels_ndebug
#endif

// detect compiler type
#if defined(__clang__)
#define wheels_compiler_clang
#elif defined(__ICC) || defined(__INTEL_COMPILER)
#define wheels_compiler_icc
#elif defined(__GNUC__) || defined(__GNUG__)
#define wheels_compiler_gcc
#elif defined(__HP_cc) || defined(__HP_aCC)
#define wheels_compiler_hpc
#elif defined(__IBMC__) || defined(__IBMCPP__)
#define wheels_compiler_xlc
#elif defined(_MSC_VER)
#define wheels_compiler_msc
#elif defined(__PGI)
#define wheels_compiler_pgc
#elif defined(__SUNPRO_C) || defined(__SUNPRO_CC)
#define wheels_compiler_sunc
#endif


#if defined(wheels_compiler_msc)
#define wheels_enable_if(B) class = std::enable_if_t<(B)>
#define wheels_enable_else_if(B) class = std::enable_if_t<(B)>, class = void
#elif defined(wheels_compiler_clang)
#define wheels_enable_if(B) bool _B = (B), class = std::enable_if_t<_B>
#define wheels_enable_else_if(B)                                               \
  bool _B = (B), class = std::enable_if_t<_B>, class = void
#endif

#define wheels_macro_cat_impl(a, b) a##b
#define wheels_macro_cat(a, b) wheels_macro_cat_impl(a, b)

#define wheels_allow_move_only(claz)                                           \
  claz(claz &&) = default;                                                     \
  claz(const claz &) = delete;                                                 \
  claz &operator=(claz &&) = default;                                          \
  claz &operator=(const claz &) = delete;