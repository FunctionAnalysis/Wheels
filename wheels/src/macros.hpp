/* * *
 * The MIT License (MIT)
 * 
 * Copyright (c) 2016 Hao Yang (yangh2007@gmail.com)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * * */

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
#else
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

#define wheels_forward(v) std::forward<decltype(v)>(v)

#ifndef wheels_no_exception
#define wheels_no_exception false
#endif

#if wheels_no_exception
#define wheels_try if(true)
#define wheels_catch(x) else
#define wheels_catch_all else
#define wheels_throw(x) (0)
#define wheels_rethrow (0)
#else
#define wheels_try try
#define wheels_catch(x) catch(x)
#define wheels_catch_all catch(...)
#define wheels_throw(x) throw x
#define wheels_rethrow throw
#endif