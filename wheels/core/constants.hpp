#pragma once

#include <array>
#include <cstdint>
#include <iostream>
#include <tuple>

#include "macros.hpp"
#include "utility.hpp"

namespace wheels {

namespace details {
// reduction helper
template <class T, T... Vals> struct _reduction {};
template <class T> struct _reduction<T> {
  static constexpr T sum = 0;
  static constexpr T prod = 1;
  static constexpr bool all = true;
  static constexpr bool any = true;
};
template <class T, T Val, T... Vals> struct _reduction<T, Val, Vals...> {
  using _rest_reduction_t = _reduction<T, Vals...>;
  static constexpr T sum = (T)(Val + _rest_reduction_t::sum);
  static constexpr T prod = (T)(Val * _rest_reduction_t::prod);
  static constexpr bool all = Val && _rest_reduction_t::all;
  static constexpr bool any = Val || _rest_reduction_t::any;
};
// element helper
template <size_t Idx, class T, T... Vals> struct _element {};
template <class T, T Val, T... Vals> struct _element<0, T, Val, Vals...> {
  static constexpr T value = Val;
};
template <size_t Idx, class T, T Val, T... Vals>
struct _element<Idx, T, Val, Vals...> {
  static constexpr T value = _element<Idx - 1, T, Vals...>::value;
};
}

// const_ints
template <class T, T... Vals> struct const_ints {
  using type = T;
  static constexpr size_t length_v = sizeof...(Vals);
  static constexpr auto length() { return const_ints<size_t, length_v>(); }

  constexpr const_ints() {}

  static constexpr auto to_array() { return std::array<T, length_v>{Vals...}; }
  static constexpr auto to_tuple() { return std::make_tuple(Vals...); }

  static constexpr T sum_v = details::_reduction<T, Vals...>::sum;
  static constexpr T prod_v = details::_reduction<T, Vals...>::prod;
  static constexpr bool all_v = details::_reduction<T, Vals...>::all;
  static constexpr bool any_v = details::_reduction<T, Vals...>::any;

  static constexpr auto sum() { return const_ints<T, sum_v>(); }
  static constexpr auto prod() { return const_ints<T, prod_v>(); }
  static constexpr auto all() { return const_ints<bool, all_v>(); }
  static constexpr auto any() { return const_ints<bool, any_v>(); }

  template <class K, K Idx>
  constexpr auto operator[](const const_ints<K, Idx> &) const {
    return const_ints<T, details::_element<Idx, T, Vals...>::value>();
  }
};

// single value
template <class T, T Val> struct const_ints<T, Val> {
  using type = T;
  static constexpr size_t length_v = 1;
  static constexpr auto length() { return const_ints<size_t, length_v>(); }

  static constexpr T value = Val;

  constexpr const_ints() {}

  static constexpr auto to_array() { return std::array<T, length_v>{Val}; }
  static constexpr auto to_tuple() { return std::make_tuple(Val); }

  static constexpr T sum_v = Val;
  static constexpr T prod_v = Val;
  static constexpr bool all_v = (bool)Val;
  static constexpr bool any_v = (bool)Val;

  static constexpr auto sum() { return const_ints<T, sum_v>(); }
  static constexpr auto prod() { return const_ints<T, prod_v>(); }
  static constexpr auto all() { return const_ints<bool, all_v>(); }
  static constexpr auto any() { return const_ints<bool, any_v>(); }

  template <class K> constexpr auto operator[](const const_ints<K, 0> &) const {
    return const_ints<T, Val>();
  }

  constexpr operator T() const { return value; }

  template <class K, K V, wheels_enable_if(V == Val)>
  constexpr operator const_ints<K, V>() const {
    return const_ints<K, V>();
  }
};

template <bool Val> using const_bool = const_ints<bool, Val>;
template <int Val> using const_int = const_ints<int, Val>;
template <size_t Val> using const_size = const_ints<size_t, Val>;
template <size_t Val> using const_index = const_ints<size_t, Val>;

using yes = const_bool<true>;
using no = const_bool<false>;

// is_const_ints
template <class T> struct is_const_ints : no {};
template <class T, T... Vals>
struct is_const_ints<const_ints<T, Vals...>> : yes {};

// is_const_int
template <class T> struct is_const_int : no {};
template <class T, T Val> struct is_const_int<const_ints<T, Val>> : yes {};

// is_int
template <class T>
struct is_int
    : const_bool<(std::is_integral<T>::value || is_const_int<T>::value)> {};

// int_traits
template <class T, bool IsNativeInt = std::is_integral<T>::value>
struct int_traits {};
template <class T> struct int_traits<T, true> {
  using type = T;
  static constexpr bool is_const_int = false;
};
template <class T, T... Val> struct int_traits<const_ints<T, Val...>, false> {
  using type = T;
  static constexpr bool is_const_int = true;
};

// conversion with std::integer_sequence
template <class T, T... Vals>
constexpr auto to_const_ints(const std::integer_sequence<T, Vals...> &) {
  return const_ints<T, Vals...>();
}
template <class T, T... Vals>
constexpr auto to_integer_sequence(const const_ints<T, Vals...> &) {
  return std::integer_sequence<T, Vals...>();
}

// conversion with std::integral_constant
template <class T, T... Vals>
constexpr auto to_const_ints(const std::integral_constant<T, Vals> &...) {
  return const_ints<T, Vals...>();
}
template <class T, T Val>
constexpr auto to_integral_constant(const const_ints<T, Val> &) {
  return std::integral_constant<T, Val>();
}

// stream
template <class T, T... Vals>
inline std::ostream &operator<<(std::ostream &os,
                                const const_ints<T, Vals...> &) {
  return print_sep_to(os, " ", Vals...);
}

namespace details {
template <class T, char... Cs> struct _parse_int {};
template <class T, char C> struct _parse_int<T, C> {
  static_assert(C >= '0' && C <= '9', "invalid character");
  enum : T { value = C - '0', magnitude = (T)1 };
};
template <class T, char C, char... Cs> struct _parse_int<T, C, Cs...> {
  static_assert(C >= '0' && C <= '9', "invalid character");
  enum : T {
    magnitude = _parse_int<T, Cs...>::magnitude * (T)10,
    rest_value = _parse_int<T, Cs...>::value,
    value = magnitude * (C - '0') + rest_value
  };
};
}

namespace literals {
// ""_c
template <char... Cs> constexpr auto operator"" _c() {
  return const_ints<int, details::_parse_int<int, Cs...>::value>();
}

// ""_uc
template <char... Cs> constexpr auto operator"" _uc() {
  return const_ints<unsigned int,
                    details::_parse_int<unsigned int, Cs...>::value>();
}

// ""_sizec
template <char... Cs> constexpr auto operator"" _sizec() {
  return const_ints<size_t, details::_parse_int<size_t, Cs...>::value>();
}

// ""_indexc
template <char... Cs> constexpr auto operator"" _indexc() {
  return const_ints<size_t, details::_parse_int<size_t, Cs...>::value>();
}

// ""_int8c
template <char... Cs> constexpr auto operator"" _int8c() {
  return const_ints<int8_t, details::_parse_int<int8_t, Cs...>::value>();
}

// ""_int16c
template <char... Cs> constexpr auto operator"" _int16c() {
  return const_ints<int16_t, details::_parse_int<int16_t, Cs...>::value>();
}

// ""_int32c
template <char... Cs> constexpr auto operator"" _int32c() {
  return const_ints<int32_t, details::_parse_int<int32_t, Cs...>::value>();
}

// ""_int64c
template <char... Cs> constexpr auto operator"" _int64c() {
  return const_ints<int64_t, details::_parse_int<int64_t, Cs...>::value>();
}

// ""_uint8c
template <char... Cs> constexpr auto operator"" _uint8c() {
  return const_ints<uint8_t, details::_parse_int<uint8_t, Cs...>::value>();
}

// ""_uint16c
template <char... Cs> constexpr auto operator"" _uint16c() {
  return const_ints<uint16_t, details::_parse_int<uint16_t, Cs...>::value>();
}

// ""_uint32c
template <char... Cs> constexpr auto operator"" _uint32c() {
  return const_ints<uint32_t, details::_parse_int<uint32_t, Cs...>::value>();
}

// ""_uint64c
template <char... Cs> constexpr auto operator"" _uint64c() {
  return const_ints<uint64_t, details::_parse_int<uint64_t, Cs...>::value>();
}

constexpr yes true_c;
constexpr no false_c;
}

// const_size_of
template <class... Ts> constexpr auto const_size_of() {
  return const_size<sizeof...(Ts)>();
}

#define WHEELS_CONST_INT_OVERLOAD_UNARY_OP(op)                                 \
  template <class T, T... Vals>                                                \
  constexpr auto operator op(const const_ints<T, Vals...> &) {                 \
    return const_ints<decltype(op std::declval<T>()), (op Vals)...>();         \
  }

WHEELS_CONST_INT_OVERLOAD_UNARY_OP(!)
WHEELS_CONST_INT_OVERLOAD_UNARY_OP(-)
WHEELS_CONST_INT_OVERLOAD_UNARY_OP(~)

#undef WHEELS_CONST_INT_OVERLOAD_UNARY_OP

#define WHEELS_CONST_INT_OVERLOAD_BINARY_OP(op)                                \
  template <class T1, T1 Val1, class T2, T2 Val2>                              \
  constexpr auto operator op(const const_ints<T1, Val1> &,                     \
                             const const_ints<T2, Val2> &) {                   \
    return const_ints<decltype(Val1 op Val2), (Val1 op Val2)>();               \
  }                                                                            \
  template <class T1, T1 Val1, class T2, T2... Val2s>                          \
  constexpr auto operator op(const const_ints<T1, Val1> &,                     \
                             const const_ints<T2, Val2s...> &) {               \
    return const_ints<decltype(Val1 op std::declval<T2>()),                    \
                      (Val1 op Val2s)...>();                                   \
  }                                                                            \
  template <class T1, T1... Val1s, class T2, T2 Val2>                          \
  constexpr auto operator op(const const_ints<T1, Val1s...> &,                 \
                             const const_ints<T2, Val2> &) {                   \
    return const_ints<decltype(std::declval<T1>() op Val2),                    \
                      (Val1s op Val2)...>();                                   \
  }                                                                            \
  template <class T1, T1... Val1s, class T2, T2... Val2s,                      \
            class = std::enable_if_t<(sizeof...(Val1s) > 1 &&                  \
                                      sizeof...(Val2s) > 1)>>                  \
  constexpr auto operator op(const const_ints<T1, Val1s...> &,                 \
                             const const_ints<T2, Val2s...> &) {               \
    static_assert(sizeof...(Val1s) == sizeof...(Val2s),                        \
                  "lengths of the two const_ints do not match");               \
    return const_ints<decltype(std::declval<T1>() op std::declval<T2>()),      \
                      (Val1s op Val2s)...>();                                  \
  }

WHEELS_CONST_INT_OVERLOAD_BINARY_OP(==)
WHEELS_CONST_INT_OVERLOAD_BINARY_OP(!=)

WHEELS_CONST_INT_OVERLOAD_BINARY_OP(>)
WHEELS_CONST_INT_OVERLOAD_BINARY_OP(<)
WHEELS_CONST_INT_OVERLOAD_BINARY_OP(>=)
WHEELS_CONST_INT_OVERLOAD_BINARY_OP(<=)

WHEELS_CONST_INT_OVERLOAD_BINARY_OP(&&)
WHEELS_CONST_INT_OVERLOAD_BINARY_OP(||)

WHEELS_CONST_INT_OVERLOAD_BINARY_OP(&)
WHEELS_CONST_INT_OVERLOAD_BINARY_OP(|)
WHEELS_CONST_INT_OVERLOAD_BINARY_OP (^)
WHEELS_CONST_INT_OVERLOAD_BINARY_OP(<<)
WHEELS_CONST_INT_OVERLOAD_BINARY_OP(>>)

WHEELS_CONST_INT_OVERLOAD_BINARY_OP(+)
WHEELS_CONST_INT_OVERLOAD_BINARY_OP(-)
WHEELS_CONST_INT_OVERLOAD_BINARY_OP(*)
WHEELS_CONST_INT_OVERLOAD_BINARY_OP(/)
WHEELS_CONST_INT_OVERLOAD_BINARY_OP(%)

#undef WHEELS_CONST_INT_OVERLOAD_BINARY_OP

// cat2
template <class T, T... Val1s, class K, K... Val2s>
constexpr auto cat2(const const_ints<T, Val1s...> &,
                    const const_ints<K, Val2s...> &) {
  using result_t = std::common_type_t<T, K>;
  return const_ints<result_t, (result_t)Val1s..., (result_t)Val2s...>();
}

// conditional
template <class T, T Val, class ThenT, class ElseT>
constexpr std::enable_if_t<Val, ThenT &&>
conditional(const const_ints<T, Val> &, ThenT &&thenv, ElseT &&elsev) {
  return static_cast<ThenT &&>(thenv);
}
template <class T, T Val, class ThenT, class ElseT, bool _B = Val>
constexpr std::enable_if_t<!_B, ElseT &&>
conditional(const const_ints<T, Val> &, ThenT &&thenv, ElseT &&elsev) {
  return static_cast<ElseT &&>(elsev);
}

namespace details {
template <class T, bool IsZero, T N, T... S>
struct _make_seq : _make_seq<T, (N - 1 == 0), N - 1, N - 1, S...> {};
template <class T, T V, T... S> struct _make_seq<T, true, V, S...> {
  using type = const_ints<T, S...>;
};

template <class T, T From, T To, T... S>
struct _make_seq_range : _make_seq_range<T, From, To - 1, To - 1, S...> {};
template <class T, T From, T... S> struct _make_seq_range<T, From, From, S...> {
  using type = const_ints<T, S...>;
};
}

// make_const_sequence
template <class T, T Size>
constexpr auto make_const_sequence(const const_ints<T, Size> &) {
  return typename details::_make_seq<T, Size == 0, Size>::type();
}

// make_const_sequence_for
template <class... Ts> constexpr auto make_const_sequence_for() {
  return make_const_sequence(const_size_of<Ts...>());
}

// make_const_range
template <class T, T From, T To>
constexpr auto make_const_range(const const_ints<T, From> &from,
                                const const_ints<T, To> &to) {
  return typename details::_make_seq_range<T, From, To>::type();
}

// repeat
namespace details {
template <class T, T S, class SeqT> struct _repeat { using type = void; };
template <class T, T S, size_t... Is>
struct _repeat<T, S, const_ints<size_t, Is...>> {
  using type = const_ints<T, always<T, S, const_index<Is>>::value...>;
};
}
template <class T, T Val, class K, K Times>
constexpr auto repeat(const const_ints<T, Val> &v,
                      const const_ints<K, Times> &times) {
  return typename details::_repeat<
      T, Val, typename details::_make_seq<size_t, Times == 0,
                                          (size_t)Times>::type>::type();
}

// count
namespace details {
template <class T, T S, T... Ss, T V>
constexpr auto _count(const const_ints<T, S, Ss...> &seq,
                      const const_ints<T, V> &v) {
  return _count(const_ints<T, Ss...>(), v);
}
template <class T, T S, T... Ss>
constexpr auto _count(const const_ints<T, S, Ss...> &seq,
                      const const_ints<T, S> &v) {
  return _count(const_ints<T, Ss...>(), v) + const_size<1>();
}
template <class T, T V>
constexpr auto _count(const const_ints<T> &, const const_ints<T, V> &) {
  return const_size<0>();
}
}
template <class T, T... S, class K, K V>
constexpr auto count(const const_ints<T, S...> &seq,
                     const const_ints<K, V> &v) {
  return details::_count(seq, const_ints<T, (T)V>());
}

// find_first_of
namespace details {
template <class T, T S, T... Ss, T V, size_t NotFoundV>
constexpr auto _find_first_of(const const_ints<T, S, Ss...> &seq,
                              const const_ints<T, V> &v,
                              const const_index<NotFoundV> &not_found) {
  return conditional(
      _find_first_of(const_ints<T, Ss...>(), v, not_found) == not_found,
      not_found,
      _find_first_of(const_ints<T, Ss...>(), v, not_found) + const_index<1>());
}
template <class T, T S, T... Ss, size_t NotFoundV>
constexpr auto _find_first_of(const const_ints<T, S, Ss...> &seq,
                              const const_ints<T, S> &v,
                              const const_index<NotFoundV> &) {
  return const_index<0>();
}
template <class T, T V, size_t NotFoundV>
constexpr auto _find_first_of(const const_ints<T> &seq,
                              const const_ints<T, V> &v,
                              const const_index<NotFoundV> &not_found) {
  return not_found;
}
}
template <class T, T S, T... Ss, class K, K V>
constexpr auto find_first_of(const const_ints<T, S, Ss...> &seq,
                             const const_ints<K, V> &v) {
  return details::_find_first_of(seq, const_ints<T, (T)V>(),
                                 const_index<1 + sizeof...(Ss)>());
}
}