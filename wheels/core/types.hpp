#pragma once

#include "constants.hpp"
#include <complex>

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
    return const_ints<bool, std::is_default_constructible<T>::value...>();
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
    return T(forward<ArgTs>(args)...);
  }

  static type_info info() { return typeid(T); }
  static const char *name() { return typeid(T).name(); }
  static const char *raw_name() { return typeid(T).raw_name(); }
};

// void
template <> struct types<void> {
  using type = void;
  static constexpr size_t length = 1;

  constexpr types() {}

  template <class K> constexpr auto operator[](const const_ints<K, 0> &) const {
    return types<T>();
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

  template <class... ArgTs> static constexpr auto construct(ArgTs &&... args) {
    return T(forward<ArgTs>(args)...);
  }

  const char *name() const { return typeid(void).name(); }
  const char *raw_name() const { return typeid(void).raw_name(); }
};

template <class... Ts> constexpr auto type_of(Ts &&... t) {
  return types<Ts &&...>();
}

// type_t
#define type_t(t) decltype(t)::type

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

// cat
template <class... T1s, class... T2s>
constexpr auto cat(const types<T1s...> &a, const types<T2s...> &b) {
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
}