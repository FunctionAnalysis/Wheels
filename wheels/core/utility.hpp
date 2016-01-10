#pragma once

#include <iostream>

#include "macros.hpp"

namespace wheels {

using std::forward;
using std::move;
template <class T> constexpr T copy(const T &t) { return t; }

// conditional for enumulating ?:
template <class ThenT, class ElseT>
constexpr decltype(auto) conditional(bool b, ThenT &&thenv, ElseT &&elsev) {
  return b ? forward<ThenT>(thenv) : forward<ElseT>(elsev);
}

// all(...)
template <class T> constexpr T &&all(T &&v) { return static_cast<T &&>(v); }
template <class T, class... Ts> constexpr auto all(T &&v, Ts &&... vs) {
  return forward<T>(v) && all(forward<Ts>(vs)...);
}

// any(...)
template <class T> constexpr T &&any(T &&v) { return static_cast<T &&>(v); }
template <class T, class... Ts> constexpr auto any(T &&v, Ts &&... vs) {
  return forward<T>(v) || any(forward<Ts>(vs)...);
}

// sum(...)
template <class T> constexpr T &&sum(T &&v) { return static_cast<T &&>(v); }
template <class T, class... Ts> constexpr auto sum(T &&v, Ts &&... vs) {
  return forward<T>(v) + sum(forward<Ts>(vs)...);
}

// prod(...)
template <class T> constexpr T &&prod(T &&v) { return static_cast<T &&>(v); }
template <class T, class... Ts> constexpr auto prod(T &&v, Ts &&... vs) {
  return forward<T>(v) * prod(forward<Ts>(vs)...);
}

// min(...)
template <class T> constexpr T &&min(T &&v) { return static_cast<T &&>(v); }
namespace details {
template <class T1, class T2> constexpr auto _min2(T1 &&a, T2 &&b) {
  return conditional(a < b, forward<T1>(a), forward<T2>(b));
}
}
template <class T, class... Ts> constexpr auto min(T &&v, Ts &&... vs) {
  return details::_min2(forward<T>(v), min(forward<Ts>(vs)...));
}

// max(...)
template <class T> constexpr T &&max(T &&v) { return static_cast<T &&>(v); }
namespace details {
template <class T1, class T2> constexpr auto _max2(T1 &&a, T2 &&b) {
  return conditional(a < b, forward<T2>(b), forward<T1>(a));
}
}
template <class T, class... Ts> constexpr auto max(T &&v, Ts &&... vs) {
  return details::_max2(forward<T>(v), max(forward<Ts>(vs)...));
}

// all_same(...)
template <class T> constexpr auto all_same(const T &a) { return yes(); }
template <class T1, class T2> constexpr auto all_same(const T1 &a, T2 &&b) {
  return a == forward<T2>(b);
}
template <class T1, class T2, class... T2s>
constexpr auto all_same(const T1 &a, T2 &&b, T2s &&... bs) {
  return a == forward<T2>(b) && all_same(a, forward<T2s>(bs)...);
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

// traverse(fun, ...)
template <class FunT, class T> constexpr void traverse(const FunT &fun, T &&v) {
  fun(forward<T>(v));
}
template <class FunT, class T, class... Ts>
constexpr void traverse(const FunT &fun, T &&v, Ts &&... vs) {
  fun(forward<T>(v));
  traverse(fun, forward<Ts>(vs)...);
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

// always
// - const value
template <class T, T Val, class... ArgTs> struct always {
  static constexpr T value = Val;
};
// - type by size_t's
template <class T, size_t... ArgIs> struct _always_t { using type = T; };
template <class T, size_t... ArgIs>
using always_t = typename _always_t<T, ArgIs...>::type;
// - type by types
template <class T, class... ArgTs> struct _always2_t { using type = T; };
template <class T, class ... ArgTs>
using always2_t = typename _always2_t<T, ArgTs...>::type;
// - functor returning const value
template <class T> struct always_f {
  constexpr always_f(const T &v) : val(v) {}
  template <class... ArgTs> constexpr const T &operator()(ArgTs &&...) const {
    return val;
  }
  T val;
};

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
}