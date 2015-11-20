#pragma once

#if !defined(NDEBUG)
#define wheels_debug
#else
#define wheels_ndebug
#endif

#if (defined _MSC_VER) || (defined __INTEL_COMPILER)
#define wheels_strong_inline __forceinline
#else
#define wheels_strong_inline inline
#endif

#if (defined _MSC_VER)
#define wheels_enable_if(B) class=std::enable_if_t<(B)>
#define wheels_enable_else_if(B) class=std::enable_if_t<(B)>,class=void
#else
#define wheels_enable_if(B) bool _B=(B),class=std::enable_if_t<_B>
#define wheels_enable_else_if(B) bool _B=(B),class=std::enable_if_t<_B>,class=void
#endif


#if (defined _PPL_H)
#define wheels_ppl
#endif

#if (defined _OPENMP)
#define wheels_openmp
#endif


#if (defined _MSC_VER)
#pragma warning(disable:4503)
#pragma warning(disable:4800)
#elif (defined __GNUG__)
#pragma GCC diagnostic ignored "-Wenum-compare"
#endif