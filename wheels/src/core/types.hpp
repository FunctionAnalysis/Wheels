#pragma once

#include <complex>

#include "types_fwd.hpp"
#include "utility_fwd.hpp"

#include "const_ints.hpp"
#include "utility.hpp"

namespace wheels {
namespace details {

// element helper
template <size_t Idx, class... Ts> struct _types_element {};
template <class T, class... Ts> struct _types_element<0, T, Ts...> {
  using type = T;
};
template <size_t Idx, class T, class... Ts>
struct _types_element<Idx, T, Ts...> {
  using type = typename _types_element<Idx - 1, Ts...>::type;
};

// is_empty helper
template <class T> struct _is_empty {
  struct _helper : T {
    int x;
  };
  static constexpr bool value = sizeof(_helper) == sizeof(int);
};
}

// types
template <class... Ts> struct types {
  static constexpr size_t length = sizeof...(Ts);
  constexpr types() {}

  template <class K, K Idx>
  constexpr auto operator[](const const_ints<K, Idx> &) const {
    return types<typename details::_types_element<Idx, Ts...>::type>();
  }

  template <class K> static constexpr auto is() {
    return const_ints<bool, std::is_same<Ts, K>::value...>();
  }
  static constexpr auto is_class() {
    return const_ints<bool, std::is_class<Ts>::value...>();
  }
  static constexpr auto is_empty() {
    return const_ints<bool, std::is_empty<Ts>::value...>();
  }
  static constexpr auto is_default_constructible() {
    return const_ints<bool, std::is_default_constructible<Ts>::value...>();
  }

  static constexpr auto decay() { return types<std::decay_t<Ts>...>(); }
};

// single type
template <class T> struct types<T> {
  using type = T;
  static constexpr size_t length = 1;

  constexpr types() {}

  template <class K> constexpr auto operator[](const const_ints<K, 0> &) const {
    return types<T>();
  }

  template <class K> static constexpr auto is() {
    return const_bool<std::is_same<T, K>::value>();
  }
  static constexpr auto is_class() {
    return const_bool<std::is_class<T>::value>();
  }
  static constexpr auto is_empty() {
    return const_bool<std::is_empty<T>::value>();
  }
  static constexpr auto is_default_constructible() {
    return const_bool<std::is_default_constructible<T>::value>();
  }

  static constexpr auto decay() { return types<std::decay_t<T>>(); }
  static constexpr auto declval() { return std::declval<T>(); }
  static constexpr T defaultv() { return T(); }
  static constexpr T zero() { return T(); }

  template <class... ArgTs> static constexpr auto construct(ArgTs &&... args) {
    return T(std::forward<ArgTs>(args)...);
  }
};

// array type
template <class T, size_t N> struct types<T[N]> {
  using type = T[N];
  static constexpr size_t length = 1;

  constexpr types() {}

  template <class K> constexpr auto operator[](const const_ints<K, 0> &) const {
    return types<T[N]>();
  }

  template <class K> static constexpr auto is() {
    return const_bool<std::is_same<T[N], K>::value>();
  }
  static constexpr auto is_class() { return no(); }
  static constexpr auto is_empty() { return no(); }
  static constexpr auto is_default_constructible() {
    return const_bool<std::is_default_constructible<T>::value>();
  }

  static constexpr auto decay() { return types<std::decay_t<T[N]>>(); }
  static constexpr auto declval() { return std::declval<T[N]>(); }
};

// void
template <> struct types<void> {
  using type = void;
  static constexpr size_t length = 1;

  constexpr types() {}

  template <class K> constexpr auto operator[](const const_ints<K, 0> &) const {
    return types<void>();
  }

  template <class K> static constexpr auto is() {
    return const_bool<std::is_same<void, K>::value>();
  }
  static constexpr auto is_class() {
    return const_bool<std::is_class<void>::value>();
  }
  static constexpr auto is_empty() {
    return const_bool<std::is_empty<void>::value>();
  }
  static constexpr auto is_default_constructible() {
    return const_bool<std::is_default_constructible<void>::value>();
  }

  static constexpr auto decay() { return types<void>(); }
};

template <class... Ts> constexpr auto type_of(Ts &&... t) {
  return types<Ts &&...>();
}

template <class... Ts>
inline std::ostream &operator<<(std::ostream &os, const types<Ts...> &) {
  os << "{";
  print_sep_to(os, ",", types<Ts>::name()...);
  return os << "}";
}

// ==
template <class... T1s, class... T2s,
          class = std::enable_if_t<sizeof...(T1s) != sizeof...(T2s)>>
constexpr no operator==(const types<T1s...> &, const types<T2s...> &) {
  return no();
}

template <class... T1s, class... T2s,
          class = std::enable_if_t<sizeof...(T1s) == sizeof...(T2s)>>
constexpr auto operator==(const types<T1s...> &, const types<T2s...> &) {
  return const_ints<bool, all(std::is_same<T1s, T2s>::value...)>();
}

template <class... T1s, class... T2s>
constexpr auto operator!=(const types<T1s...> &a, const types<T2s...> &b) {
  return !(a == b);
}

// cat2
template <class... T1s, class... T2s>
constexpr auto cat2(const types<T1s...> &a, const types<T2s...> &b) {
  return types<T1s..., T2s...>();
}

namespace details {
template <size_t Bytes> struct _int_of {};
template <size_t Bytes> struct _uint_of {};

template <> struct _int_of<1> { using type = int8_t; };
template <> struct _uint_of<1> { using type = uint8_t; };
template <> struct _int_of<2> { using type = int16_t; };
template <> struct _uint_of<2> { using type = uint16_t; };
template <> struct _int_of<4> { using type = int32_t; };
template <> struct _uint_of<4> { using type = uint32_t; };
template <> struct _int_of<8> { using type = int64_t; };
template <> struct _uint_of<8> { using type = uint64_t; };
}

// int_type_of_bytes
template <class T, T... Bs>
constexpr auto int_type_of_bytes(const const_ints<T, Bs...> &) {
  return types<typename details::_int_of<Bs>::type...>();
}

// uint_type_of_bytes
template <class T, T... Bs>
constexpr auto uint_type_of_bytes(const const_ints<T, Bs...> &) {
  return types<typename details::_uint_of<Bs>::type...>();
}

// is_complex
template <class T> struct is_complex : no {};
template <class T> struct is_complex<std::complex<T>> : yes {};

// real_component
template <class T> struct real_component { using type = T; };
template <class T> struct real_component<std::complex<T>> { using type = T; };

// is_initializer_list
template <class T> struct is_initializer_list : no {};
template <class T>
struct is_initializer_list<std::initializer_list<T>> : yes {};

// is_character
template <class T> struct is_character : no {};
template <> struct is_character<char> : yes {};
template <> struct is_character<wchar_t> : yes {};
template <> struct is_character<char16_t> : yes {};
template <> struct is_character<char32_t> : yes {};

// is_zero
template <class T>
std::enable_if_t<std::is_scalar<T>::value, bool> is_zero(const T &v) {
  return v == 0;
}
template <class T> bool is_zero(const std::complex<T> &v) {
  return is_zero(v.real()) && is_zero(v.imag());
}

// type_restrict
namespace details {
template <class RestrictT, class T>
constexpr T &&_type_restrict(const RestrictT &, T &&t) {
  return static_cast<T &&>(t);
}
}
template <class RestrictT, class T>
constexpr auto type_restrict(T &&t)
    -> decltype(details::_type_restrict<RestrictT>(t, std::forward<T>(t))) {
  return details::_type_restrict<RestrictT>(t, std::forward<T>(t));
}
}
