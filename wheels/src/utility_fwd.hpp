#pragma once

#include <iostream>

namespace wheels {

template <class... Ts> constexpr auto as_tuple(Ts &&... ts);

// conditional for enumulating ?:
template <class ThenT, class ElseT>
constexpr auto conditional(bool b, ThenT &&thenv, ElseT &&elsev);

// all
template <class T> constexpr T &&all(T &&v);
template <class T, class... Ts> constexpr auto all(T &&v, Ts &&... vs);

// any(...)
template <class T> constexpr T &&any(T &&v);
template <class T, class... Ts> constexpr auto any(T &&v, Ts &&... vs);

// sum(...)
template <class T> constexpr T &&sum(T &&v);
template <class T, class... Ts> constexpr auto sum(T &&v, Ts &&... vs);

// prod(...)
template <class T> constexpr T &&prod(T &&v);
template <class T, class... Ts> constexpr auto prod(T &&v, Ts &&... vs);

// min(...)
template <class T> constexpr T &&min(T &&v);
template <class T, class... Ts> constexpr auto min(T &&v, Ts &&... vs);

// max(...)
template <class T> constexpr T &&max(T &&v);
template <class T, class... Ts> constexpr auto max(T &&v, Ts &&... vs);

// all_same(...)
template <class T> constexpr auto all_same(const T &a);
template <class T1, class T2> constexpr auto all_same(const T1 &a, T2 &&b);
template <class T1, class T2, class... T2s>
constexpr auto all_same(const T1 &a, T2 &&b, T2s &&... bs);

// all_different(...)
template <class T> constexpr auto all_different(const T &a);
template <class T1, class T2>
constexpr auto all_different(const T1 &a, const T2 &b);
template <class T1, class T2, class... T2s>
constexpr auto all_different(const T1 &a, const T2 &b, const T2s &... bs);

// cat(...)
// cat2(a, b) required
template <class T> constexpr T &&cat(T &&a);
template <class T, class... Ts> constexpr auto cat(T &&v, Ts &&... vs);

// traverse(fun, ...)
template <class FunT> constexpr void traverse(FunT fun);
template <class FunT, class T, class... Ts>
constexpr void traverse(FunT fun, T &&v, Ts &&... vs);
// make_ordered_pair
template <class T1, class T2>
constexpr decltype(auto) make_ordered_pair(T1 &&a, T2 &&b);

// close_bounded [lb, ub]
template <class T, class LowBT, class UpBT>
constexpr decltype(auto) close_bounded(T &&v, LowBT &&lb, UpBT &&ub);

// is_between [lb, ub)
template <class T, class LowBT, class UpBT>
constexpr decltype(auto) is_between(T &&v, LowBT &&lb, UpBT &&ub);

// right_open_wrapped [lb, ub)
template <class T>
constexpr T right_open_wrapped(const T &v, const T &lb, const T &ub);

// always
// - const value
template <class T, T Val, class... ArgTs> struct always {
  static constexpr T value = Val;
};
// - type by size_t's
namespace details {
template <class T, size_t... ArgIs> struct _always_t { using type = T; };
}
template <class T, size_t... ArgIs>
using always_t = typename details::_always_t<T, ArgIs...>::type;
// - type by types
namespace details {
template <class T, class... ArgTs> struct _always2_t { using type = T; };
}
template <class T, class... ArgTs>
using always2_t = typename details::_always2_t<T, ArgTs...>::type;
// - functor returning const value
template <class T> struct always_f {
  constexpr always_f(const T &v) : val(v) {}
  template <class... ArgTs> constexpr const T &operator()(ArgTs &&...) const {
    return val;
  }
  T val;
};

// void_t
template <class ...> using void_t = void;

// print_to
inline std::ostream &print_to(std::ostream &os);
template <class T, class... Ts>
inline std::ostream &print_to(std::ostream &os, const T &arg,
                              const Ts &... args);
// print_sep_to
template <class SepT>
inline std::ostream &print_sep_to(std::ostream &os, SepT &&sep);
template <class SepT, class T>
inline std::ostream &print_sep_to(std::ostream &os, SepT &&sep, const T &arg);
template <class SepT, class T, class... Ts>
inline std::ostream &print_sep_to(std::ostream &os, SepT &&sep, const T &arg,
                                  const Ts &... args);

// print
template <class... Ts> inline std::ostream &print(const Ts &... args);
// print_sep
template <class SepT, class... Ts>
inline std::ostream &print_sep(SepT &&sep, const Ts &... args);
// println
template <class... Ts> inline std::ostream &println(const Ts &... args);
// print_sep
template <class SepT, class... Ts>
inline std::ostream &println_sep(SepT &&sep, const Ts &... args);
}
