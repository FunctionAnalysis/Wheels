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

#include <iostream>
#include <random>
#include <tuple>
#include <complex>

#include "utility_fwd.hpp"

#include "const_ints.hpp"
#include "macros.hpp"

namespace wheels {

template <class... Ts> constexpr auto as_tuple(Ts &&... ts) {
  return std::tuple<Ts...>(std::forward<Ts>(ts)...);
}

using ignore_t = decltype(std::ignore);

// conditional for enumulating ?:
template <class ThenT, class ElseT>
constexpr auto conditional(bool b, ThenT &&thenv, ElseT &&elsev) {
  return b ? std::forward<ThenT>(thenv) : std::forward<ElseT>(elsev);
}

// all(...)
template <class T> constexpr T &&all(T &&v) { return static_cast<T &&>(v); }
template <class T, class... Ts> constexpr auto all(T &&v, Ts &&... vs) {
  return std::forward<T>(v) && all(std::forward<Ts>(vs)...);
}

// any(...)
template <class T> constexpr T &&any(T &&v) { return static_cast<T &&>(v); }
template <class T, class... Ts> constexpr auto any(T &&v, Ts &&... vs) {
  return std::forward<T>(v) || any(std::forward<Ts>(vs)...);
}

// sum(...)
template <class T> constexpr T &&sum(T &&v) { return static_cast<T &&>(v); }
template <class T, class... Ts> constexpr auto sum(T &&v, Ts &&... vs) {
  return std::forward<T>(v) + sum(std::forward<Ts>(vs)...);
}

// prod(...)
template <class T> constexpr T &&prod(T &&v) { return static_cast<T &&>(v); }
template <class T, class... Ts> constexpr auto prod(T &&v, Ts &&... vs) {
  return std::forward<T>(v) * prod(std::forward<Ts>(vs)...);
}

// min(...)
template <class T> constexpr T &&min(T &&v) { return static_cast<T &&>(v); }
namespace details {
template <class T1, class T2> constexpr auto _min2(T1 &&a, T2 &&b) {
  return conditional(a < b, std::forward<T1>(a), std::forward<T2>(b));
}
}
template <class T, class... Ts> constexpr auto min(T &&v, Ts &&... vs) {
  return details::_min2(std::forward<T>(v), min(std::forward<Ts>(vs)...));
}

// max(...)
template <class T> constexpr T &&max(T &&v) { return static_cast<T &&>(v); }
namespace details {
template <class T1, class T2> constexpr auto _max2(T1 &&a, T2 &&b) {
  return conditional(a < b, std::forward<T2>(b), std::forward<T1>(a));
}
}
template <class T, class... Ts> constexpr auto max(T &&v, Ts &&... vs) {
  return details::_max2(std::forward<T>(v), max(std::forward<Ts>(vs)...));
}

// all_same(...)
template <class T> constexpr auto all_same(const T &a) { return yes(); }
template <class T1, class T2> constexpr auto all_same(const T1 &a, T2 &&b) {
  return a == std::forward<T2>(b);
}
template <class T1, class T2, class... T2s>
constexpr auto all_same(const T1 &a, T2 &&b, T2s &&... bs) {
  return a == std::forward<T2>(b) && all_same(a, std::forward<T2s>(bs)...);
}

// all_different(...)
template <class T> constexpr auto all_different(const T &a) { return yes(); }
template <class T1, class T2>
constexpr auto all_different(const T1 &a, const T2 &b) {
  return a != b;
}
template <class T1, class T2, class... T2s>
constexpr auto all_different(const T1 &a, const T2 &b, const T2s &... bs) {
  return all(a != b, a != bs...) && all_different(b, bs...);
}

// cat(...)
// cat2(a, b) required
template <class T> constexpr T &&cat(T &&a) { return static_cast<T &&>(a); }
template <class T, class... Ts> constexpr auto cat(T &&v, Ts &&... vs) {
  return cat2(std::forward<T>(v), cat(std::forward<Ts>(vs)...));
}

// traverse(fun, ...)
template <class FunT> constexpr void traverse(FunT fun) {}
template <class FunT, class T, class... Ts>
constexpr void traverse(FunT fun, T &&v, Ts &&... vs) {
  fun(std::forward<T>(v));
  traverse(fun, std::forward<Ts>(vs)...);
}

// make_ordered_pair
template <class T1, class T2>
constexpr decltype(auto) make_ordered_pair(T1 &&a, T2 &&b) {
  return conditional(a < b, std::make_pair(a, b), std::make_pair(b, a));
}

// close_bounded [lb, ub]
template <class T, class LowBT, class UpBT>
constexpr decltype(auto) close_bounded(T &&v, LowBT &&lb, UpBT &&ub) {
  return conditional(v < lb, lb, conditional(v < ub, v, ub));
}

// is_between [lb, ub)
template <class T, class LowBT, class UpBT>
constexpr decltype(auto) is_between(T &&v, LowBT &&lb, UpBT &&ub) {
  return !(v < lb) && v < ub;
}

// right_open_wrapped [lb, ub)
namespace details {
template <class T>
constexpr T _right_open_wrapped(const T &input, const T &low, const T &high,
                                const std::false_type &isint) {
  if (low >= high)
    return input;
  if (low <= input && input < high)
    return input;
  const auto sz = high - low;
  auto result = input - int((input - low) / sz) * sz + (input < low ? sz : 0);
  return result == high ? low : result;
}
template <class T>
constexpr T _right_open_wrapped(const T &input, const T &low, const T &high,
                                const std::true_type &isint) {
  if (low >= high)
    return input;
  if (low <= input && input < high)
    return input;
  const auto sz = high - low;
  auto result = (input - low) % sz + low + (input < low ? sz : 0);
  return result == high ? low : result;
}
}
template <class T>
constexpr T right_open_wrapped(const T &v, const T &lb, const T &ub) {
  return details::_right_open_wrapped(v, lb, ub, std::is_integral<T>());
}

// print_to
inline std::ostream &print_to(std::ostream &os) { return os; }
template <class T, class... Ts>
inline std::ostream &print_to(std::ostream &os, const T &arg,
                              const Ts &... args) {
  os << arg;
  return print_to(os, args...);
}
// print_sep_to
template <class SepT>
inline std::ostream &print_sep_to(std::ostream &os, SepT &&sep) {
  return os;
}
template <class SepT, class T>
inline std::ostream &print_sep_to(std::ostream &os, SepT &&sep, const T &arg) {
  return os << arg;
}
template <class SepT, class T, class... Ts>
inline std::ostream &print_sep_to(std::ostream &os, SepT &&sep, const T &arg,
                                  const Ts &... args) {
  os << arg << sep;
  return print_sep_to(os, sep, args...);
}

// print
template <class... Ts> inline std::ostream &print(const Ts &... args) {
  return print_to(std::cout, args...);
}
// print_sep
template <class SepT, class... Ts>
inline std::ostream &print_sep(SepT &&sep, const Ts &... args) {
  return print_sep_to(std::cout, sep, args...);
}
// println
template <class... Ts> inline std::ostream &println(const Ts &... args) {
  return print(args...) << std::endl;
}
// print_sep
template <class SepT, class... Ts>
inline std::ostream &println_sep(SepT &&sep, const Ts &... args) {
  return print_sep_to(std::cout, sep, args...) << std::endl;
}

// randomize integral scalar
template <class T, class RNG>
std::enable_if_t<std::is_integral<T>::value> randomize(T &v, RNG &rng) {
  static const T minv = std::numeric_limits<T>::lowest();
  static const T maxv = std::numeric_limits<T>::max();
  v = static_cast<T>(std::uniform_int_distribution<std::intmax_t>(
      (std::intmax_t)minv, (std::intmax_t)maxv)(rng));
}

// randomize float scalar
template <class T, class RNG>
std::enable_if_t<std::is_floating_point<T>::value> randomize(T &v, RNG &rng) {
  static const T minv = T(-1.0);
  static const T maxv = T(1.0);
  v = std::uniform_real_distribution<T>(minv, maxv)(rng);
}

// randomize complex numbers
template <class T, class RNG> void randomize(std::complex<T> &v, RNG &rng) {
  T real, imag;
  randomize(real, rng);
  randomize(imag, rng);
  v.real(real);
  v.imag(imag);
}
}