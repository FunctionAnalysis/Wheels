#pragma once

namespace wheels {

// tuplize
template <class T> constexpr auto tuplize(T &&data);

// traverse_fields
template <class T, class FunT>
constexpr void traverse_fields(T &&data, FunT fun);

// randomize_fields
template <class T, class RNG> inline void randomize_fields(T &data, RNG &rng);

// any_of_fields
template <class T, class CheckFunT>
constexpr bool any_of_fields(const T &data, CheckFunT checker);

// none_of_fields
template <class T, class CheckFunT>
constexpr bool none_of_fields(const T &data, CheckFunT checker);

// all_of_fields
template <class T, class CheckFunT>
constexpr bool all_of_fields(const T &data, CheckFunT checker);

// comparable
template <class T, class Kind = T> struct comparable;
}
