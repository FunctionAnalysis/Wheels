#pragma once

#include <deque>
#include <list>
#include <tuple>
#include <type_traits>
#include <vector>

#include "object_fwd.hpp"

namespace wheels {
namespace category {

// object
template <class T> struct object {
  constexpr const T &derived() const & { return static_cast<const T &>(*this); }
  T &derived() & { return static_cast<T &>(*this); }
  T &&derived() && { return static_cast<T &&>(*this); }
};

template <class T> constexpr const T &identify_impl(const object<T> &o) {
  return o.derived();
}

// other
template <class T> struct other {};

template <class T> struct scalar : other<T> {};
template <class T> struct arithmetic : scalar<T> {};

// integral
template <class T> struct integral : arithmetic<T> {};
template <class T, class = std::enable_if_t<std::is_integral<T>::value>>
constexpr const integral<T> &identify_impl(const T &t) {
  return reinterpret_cast<const integral<T> &>(t);
}

// floating point
template <class T> struct floating_point : arithmetic<T> {};
template <class T, class = std::enable_if_t<std::is_floating_point<T>::value>>
constexpr const floating_point<T> &identify_impl(const T &t) {
  return reinterpret_cast<const floating_point<T> &>(t);
}

// std_container
template <class T> struct std_container : other<T> {};
// vector
template <class T, class AllocT>
constexpr const std_container<std::vector<T, AllocT>> &
identify_impl(const std::vector<T, AllocT> &t) {
  return reinterpret_cast<const std_container<std::vector<T, AllocT>> &>(t);
}
// list
template <class T, class AllocT>
constexpr const std_container<std::list<T, AllocT>> &
identify_impl(const std::list<T, AllocT> &t) {
  return reinterpret_cast<const std_container<std::list<T, AllocT>> &>(t);
}
// deque
template <class T, class AllocT>
constexpr const std_container<std::deque<T, AllocT>> &
identify_impl(const std::deque<T, AllocT> &t) {
  return reinterpret_cast<const std_container<std::deque<T, AllocT>> &>(t);
}
// array
template <class T, size_t N>
constexpr const std_container<std::array<T, N>> &
identify_impl(const std::array<T, N> &t) {
  return reinterpret_cast<const std_container<std::array<T, N>> &>(t);
}
// raw array
template <class T, size_t N>
constexpr const std_container<T[N]> &identify_impl(T const (&t)[N]) {
  return reinterpret_cast<const std_container<T[N]> &>(t);
}

// std_tuplelike
template <class T> struct std_tuplelike : other<T> {};
// tuple
template <class... Ts>
constexpr const std_tuplelike<std::tuple<Ts...>> &
identify_impl(const std::tuple<Ts...> &t) {
  return reinterpret_cast<const std_tuplelike<std::tuple<Ts...>> &>(t);
}
// pair
template <class T1, class T2>
constexpr const std_tuplelike<std::pair<T1, T2>> &
identify_impl(const std::pair<T1, T2> &t) {
  return reinterpret_cast<const std_tuplelike<std::pair<T1, T2>> &>(t);
}

// identify_impl
template <class T,
          class = std::enable_if_t<
              !std::is_convertible<const T &, const object<T> &>::value &&
              !std::is_integral<T>::value && !std::is_floating_point<T>::value>>
constexpr const other<T> &identify_impl(const T &t) {
  return *reinterpret_cast<const other<T> *>(nullptr);
}

// identify
template <class T> constexpr decltype(auto) identify(const T &t) {
  return identify_impl(t);
}
}
}