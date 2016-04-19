#pragma once

#include <type_traits>

namespace wheels {

// tuplize
template <class T> constexpr auto tuplize(T &&data);

// traverse_fields
template <class T, class FunT>
constexpr void traverse_fields(T &&data, FunT fun);

// randomize_fields
template <class T, class RNG> inline void randomize_fields(T &data, RNG &rng);

// default_boolean_checker
struct default_boolean_checker {
  template <class T> constexpr bool operator()(T &&t) const {
    return (bool)std::forward<T>(t);
  }
};

// any_of_fields
template <class T, class CheckFunT = default_boolean_checker>
constexpr bool any_of_fields(const T &data, CheckFunT checker = CheckFunT());

// none_of_fields
template <class T, class CheckFunT = default_boolean_checker>
constexpr bool none_of_fields(const T &data, CheckFunT checker = CheckFunT());

// all_of_fields
template <class T, class CheckFunT = default_boolean_checker>
constexpr bool all_of_fields(const T &data, CheckFunT checker = CheckFunT());

// comparable
template <class T, class Kind = T> struct comparable;
}
