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

#include <algorithm>

#include "overloads_fwd.hpp"

#include "const_ints.hpp"
#include "types.hpp"
#include "utility.hpp"
#include "what.hpp"

namespace wheels {
// func_base
template <class OpT> struct func_base : object_base<OpT> {
  const OpT &derived() const { return static_cast<const OpT &>(*this); }
};

// overload_as (default implementation)
template <class OpT, class... CatTs>
inline auto overload_as(const func_base<OpT> &, const proxy_base<CatTs> &...) {}

#define WHEELS_OVERLOAD_UNARY_OP(op, name)                                     \
  struct unary_op_##name : func_base<unary_op_##name> {                        \
    constexpr unary_op_##name() {}                                             \
    template <class TT> constexpr decltype(auto) operator()(TT &&v) const {    \
      return (op std::forward<TT>(v));                                         \
    }                                                                          \
  };                                                                           \
  template <class T>                                                           \
  constexpr auto operator op(T &&v)->decltype(                                 \
      overload_as(unary_op_##name(), ::wheels::what(v))(std::forward<T>(v))) { \
    return overload_as(unary_op_##name(),                                      \
                       ::wheels::what(v))(std::forward<T>(v));                 \
  }

WHEELS_OVERLOAD_UNARY_OP(-, minus)
WHEELS_OVERLOAD_UNARY_OP(!, not)
WHEELS_OVERLOAD_UNARY_OP(~, bitwise_not)
#undef WHEELS_OVERLOAD_UNARY_OP

// WHEELS_OVERLOAD_BINARY_OP
#define WHEELS_OVERLOAD_BINARY_OP(op, name)                                    \
  struct binary_op_##name : func_base<binary_op_##name> {                      \
    constexpr binary_op_##name() {}                                            \
    template <class TT1, class TT2>                                            \
    constexpr decltype(auto) operator()(TT1 &&v1, TT2 &&v2) const {            \
      return (std::forward<TT1>(v1) op std::forward<TT2>(v2));                 \
    }                                                                          \
  };                                                                           \
  template <class T1, class T2>                                                \
  constexpr auto operator op(T1 &&v1, T2 &&v2)                                 \
      ->decltype(overload_as(binary_op_##name(), ::wheels::what(v1),           \
                             ::wheels::what(v2))(std::forward<T1>(v1),         \
                                                 std::forward<T2>(v2))) {      \
    return overload_as(binary_op_##name(), ::wheels::what(v1),                 \
                       ::wheels::what(v2))(std::forward<T1>(v1),               \
                                           std::forward<T2>(v2));              \
  }

WHEELS_OVERLOAD_BINARY_OP(+, plus)
WHEELS_OVERLOAD_BINARY_OP(-, minus)
WHEELS_OVERLOAD_BINARY_OP(*, mul)
WHEELS_OVERLOAD_BINARY_OP(/, div)
WHEELS_OVERLOAD_BINARY_OP(%, mod)

WHEELS_OVERLOAD_BINARY_OP(==, eq)
WHEELS_OVERLOAD_BINARY_OP(!=, neq)
WHEELS_OVERLOAD_BINARY_OP(<, lt)
WHEELS_OVERLOAD_BINARY_OP(<=, lte)
WHEELS_OVERLOAD_BINARY_OP(>, gt)
WHEELS_OVERLOAD_BINARY_OP(>=, gte)

WHEELS_OVERLOAD_BINARY_OP(&&, and)
WHEELS_OVERLOAD_BINARY_OP(||, or)
WHEELS_OVERLOAD_BINARY_OP(&, bitwise_and)
WHEELS_OVERLOAD_BINARY_OP(|, bitwise_or)
WHEELS_OVERLOAD_BINARY_OP (^, bitwise_xor)

#undef WHEELS_OVERLOAD_BINARY_OP

// WHEELS_OVERLOAD_STD_UNARY_FUNC
// WHEELS_OVERLOAD_STD_BINARY_FUNC
#define WHEELS_OVERLOAD_STD_UNARY_FUNC(name)                                   \
  struct std_func_##name : func_base<std_func_##name> {                        \
    constexpr std_func_##name() {}                                             \
    template <class TT> constexpr decltype(auto) operator()(TT &&v) const {    \
      using std::name;                                                         \
      return name(std::forward<TT>(v));                                        \
    }                                                                          \
  };                                                                           \
  template <class T>                                                           \
  constexpr auto name(T &&v)->decltype(                                        \
      overload_as(std_func_##name(), ::wheels::what(v))(std::forward<T>(v))) { \
    return overload_as(std_func_##name(),                                      \
                       ::wheels::what(v))(std::forward<T>(v));                 \
  }

#define WHEELS_OVERLOAD_STD_BINARY_FUNC(name)                                  \
  struct std_func_##name : func_base<std_func_##name> {                        \
    constexpr std_func_##name() {}                                             \
    template <class ArgT1, class ArgT2>                                        \
    constexpr auto operator()(ArgT1 &&v1, ArgT2 &&v2) const {                  \
      using std::name;                                                         \
      return name(std::forward<ArgT1>(v1), std::forward<ArgT2>(v2));           \
    }                                                                          \
  };                                                                           \
  template <class T1, class T2>                                                \
  constexpr auto name(T1 &&t1, T2 &&t2)                                        \
      ->decltype(overload_as(std_func_##name(), ::wheels::what(t1),            \
                             ::wheels::what(t2))(std::forward<T1>(t1),         \
                                                 std::forward<T2>(t2))) {      \
    return overload_as(std_func_##name(), ::wheels::what(t1),                  \
                       ::wheels::what(t2))(std::forward<T1>(t1),               \
                                           std::forward<T2>(t2));              \
  }

WHEELS_OVERLOAD_STD_UNARY_FUNC(sin)
WHEELS_OVERLOAD_STD_UNARY_FUNC(sinh)
WHEELS_OVERLOAD_STD_UNARY_FUNC(asin)
WHEELS_OVERLOAD_STD_UNARY_FUNC(asinh)
WHEELS_OVERLOAD_STD_UNARY_FUNC(cos)
WHEELS_OVERLOAD_STD_UNARY_FUNC(cosh)
WHEELS_OVERLOAD_STD_UNARY_FUNC(acos)
WHEELS_OVERLOAD_STD_UNARY_FUNC(acosh)
WHEELS_OVERLOAD_STD_UNARY_FUNC(tan)
WHEELS_OVERLOAD_STD_UNARY_FUNC(tanh)
WHEELS_OVERLOAD_STD_UNARY_FUNC(atan)
WHEELS_OVERLOAD_STD_UNARY_FUNC(atanh)
WHEELS_OVERLOAD_STD_UNARY_FUNC(log)
WHEELS_OVERLOAD_STD_UNARY_FUNC(log2)
WHEELS_OVERLOAD_STD_UNARY_FUNC(log10)
WHEELS_OVERLOAD_STD_UNARY_FUNC(exp)
WHEELS_OVERLOAD_STD_UNARY_FUNC(exp2)
WHEELS_OVERLOAD_STD_UNARY_FUNC(ceil)
WHEELS_OVERLOAD_STD_UNARY_FUNC(floor)
WHEELS_OVERLOAD_STD_UNARY_FUNC(round)
WHEELS_OVERLOAD_STD_UNARY_FUNC(isinf)
WHEELS_OVERLOAD_STD_UNARY_FUNC(isfinite)
WHEELS_OVERLOAD_STD_UNARY_FUNC(isnan)
WHEELS_OVERLOAD_STD_UNARY_FUNC(abs)

WHEELS_OVERLOAD_STD_BINARY_FUNC(atan2)
WHEELS_OVERLOAD_STD_BINARY_FUNC(pow)
WHEELS_OVERLOAD_STD_BINARY_FUNC(min)
WHEELS_OVERLOAD_STD_BINARY_FUNC(max)

#undef WHEELS_OVERLOAD_STD_UNARY_FUNC
#undef WHEELS_OVERLOAD_STD_BINARY_FUNC
}