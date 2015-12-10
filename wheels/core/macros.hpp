#pragma once

#if !defined(NDEBUG)
#define wheels_debug
#else
#define wheels_ndebug
#endif

#if (defined _MSC_VER)
#define wheels_enable_if(B) class = std::enable_if_t<(B)>
#define wheels_enable_else_if(B) class = std::enable_if_t<(B)>, class = void
#else
#define wheels_enable_if(B) bool _B = (B), class = std::enable_if_t<_B>
#define wheels_enable_else_if(B)                                               \
  bool _B = (B), class = std::enable_if_t<_B>, class = void
#endif

#if (defined _MSC_VER)
#pragma warning(disable : 4503)
#pragma warning(disable : 4800)
#elif (defined __GNUG__)
#pragma GCC diagnostic ignored "-Wenum-compare"
#endif

#define wheels_distinguish_1 class = void
#define wheels_distinguish_2 wheels_distinguish_1, wheels_distinguish_1
#define wheels_distinguish_3 wheels_distinguish_2, wheels_distinguish_1
#define wheels_distinguish_4 wheels_distinguish_2, wheels_distinguish_2
#define wheels_distinguish_5 wheels_distinguish_4, wheels_distinguish_1
#define wheels_distinguish_6 wheels_distinguish_3, wheels_distinguish_3