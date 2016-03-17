#pragma once

#include <type_traits>

namespace wheels {
namespace kinds {

// object
template <class T> struct object {
  constexpr const T &derived() const & { return static_cast<const T &>(*this); }
  T &derived() & { return static_cast<T &>(*this); }
  T &&derived() && { return static_cast<T &&>(*this); }
};

template <class T> constexpr const T &identify(const object<T> &o) {
  return o.derived();
}

// other
template <class T> struct other {
  constexpr const T &derived() const & {
    return reinterpret_cast<const T &>(*this);
  }
  T &derived() & { return reinterpret_cast<T &>(*this); }
};

template <class T> struct scalar : other<T> {};

template <class T> struct enumeration : scalar<T> {};
template <class T> struct pointer : scalar<T> {};
template <class T> struct member_pointer : scalar<T> {};
template <class T> struct null_pointer : scalar<T> {};

template <class T> struct arithmetic : scalar<T> {};

template <class T> struct integral : arithmetic<T> {};
template <class T, class = std::enable_if_t<std::is_integral<T>::value>>
constexpr const integral<T> &identify(const T &t) {
  return reinterpret_cast<const integral<T> &>(t);
}

template <class T> struct floating_point : arithmetic<T> {};
template <class T, class = std::enable_if_t<std::is_floating_point<T>::value>>
constexpr const floating_point<T> &identify(const T &t) {
  return reinterpret_cast<const floating_point<T> &>(t);
}

template <class T,
          class = std::enable_if_t<
              !std::is_convertible<const T &, const object<T> &>::value &&
              !std::is_integral<T>::value && !std::is_floating_point<T>::value>>
constexpr const other<T> &identify(const T &t) {
  return *reinterpret_cast<const other<T> *>(&t);
}
}
}