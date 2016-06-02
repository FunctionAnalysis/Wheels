#pragma once

#include <deque>
#include <list>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "what_fwd.hpp"

namespace wheels {

// object_base
template <class T> struct object_base {
  constexpr const T &derived() const & { return static_cast<const T &>(*this); }
  T &derived() & { return static_cast<T &>(*this); }
  T &&derived() && { return static_cast<T &&>(*this); }
};
template <class T> constexpr const T &it_is(const object_base<T> &o) {
  return o.derived();
}

namespace details {
template <class T> constexpr const T &_gen_ref() { return *((const T *)(0)); }
}

// proxy_base
template <class T> struct proxy_base {};

template <class T> struct scalar : proxy_base<T> {};
template <class T> struct arithmetic : scalar<T> {};

// integral
template <class T> struct integral : arithmetic<T> {};
template <class T, class = std::enable_if_t<std::is_integral<T>::value>>
constexpr const integral<T> &it_is(const T &t) {
  return details::_gen_ref<integral<T>>();
}

// floating point
template <class T> struct floating_point : arithmetic<T> {};
template <class T, class = std::enable_if_t<std::is_floating_point<T>::value>>
constexpr const floating_point<T> &it_is(const T &t) {
  return details::_gen_ref<floating_point<T>>();
}

// enumerate
template <class T> struct enumerate : scalar<T> {};
template <class T, class = std::enable_if_t<std::is_enum<T>::value>>
constexpr const enumerate<T> &it_is(const T &t) {
  return details::_gen_ref<enumerate<T>>();
}

// pointer
template <class T> struct pointer : scalar<T> {};
template <class T, class = std::enable_if_t<std::is_pointer<T>::value>>
constexpr const pointer<T> &it_is(const T &t) {
  return details::_gen_ref<pointer<T>>();
}

// null_pointer
template <class T> struct null_pointer : proxy_base<T> {};
inline const null_pointer<nullptr_t> &it_is(nullptr_t) {
  return details::_gen_ref<null_pointer<nullptr_t>>();
}

// std_container
template <class T> struct std_container : proxy_base<T> {};
template <class T> struct std_sequence : std_container<T> {};
template <class T> struct is_std_sequence : no {};
template <class T> struct is_std_container : is_std_sequence<T> {};

// set
template <class T, class CompT, class AllocT>
constexpr const std_container<std::set<T, CompT, AllocT>> &
it_is(const std::set<T, CompT, AllocT> &t) {
  return details::_gen_ref<std_container<std::set<T, CompT, AllocT>>>();
}
template <class T, class CompT, class AllocT>
struct is_std_container<std::set<T, CompT, AllocT>> : yes {};

// unordered_set
template <class T, class HashT, class EqT, class AllocT>
constexpr const std_container<std::unordered_set<T, HashT, EqT, AllocT>> &
it_is(const std::unordered_set<T, HashT, EqT, AllocT> &t) {
  return std::declval<
      const std_container<std::unordered_set<T, HashT, EqT, AllocT>>>();
}
template <class T, class HashT, class EqT, class AllocT>
struct is_std_container<std::unordered_set<T, HashT, EqT, AllocT>> : yes {};

// vector
template <class T, class AllocT>
constexpr const std_sequence<std::vector<T, AllocT>> &
it_is(const std::vector<T, AllocT> &t) {
  return details::_gen_ref<std_sequence<std::vector<T, AllocT>>>();
}
template <class T, class AllocT>
struct is_std_sequence<std::vector<T, AllocT>> : yes {};

// list
template <class T, class AllocT>
constexpr const std_sequence<std::list<T, AllocT>> &
it_is(const std::list<T, AllocT> &t) {
  return details::_gen_ref<std_sequence<std::list<T, AllocT>>>();
}
template <class T, class AllocT>
struct is_std_sequence<std::list<T, AllocT>> : yes {};

// deque
template <class T, class AllocT>
constexpr const std_sequence<std::deque<T, AllocT>> &
it_is(const std::deque<T, AllocT> &t) {
  return details::_gen_ref<std_sequence<std::deque<T, AllocT>>>();
}
template <class T, class AllocT>
struct is_std_sequence<std::deque<T, AllocT>> : yes {};

// array
template <class T, size_t N>
constexpr const std_sequence<std::array<T, N>> &
it_is(const std::array<T, N> &t) {
  return details::_gen_ref<std_sequence<std::array<T, N>>>();
}
template <class T, size_t N> struct is_std_sequence<std::array<T, N>> : yes {};

// raw array
template <class T, size_t N>
constexpr const std_sequence<T[N]> &it_is(T const (&t)[N]) {
  return details::_gen_ref<std_sequence<T[N]>>();
}
template <class T, size_t N> struct is_std_sequence<T[N]> : yes {};

// std_tuplelike
template <class T> struct std_tuplelike : proxy_base<T> {};
template <class T> struct is_std_tuplelike : no {};

// tuple
template <class... Ts>
constexpr const std_tuplelike<std::tuple<Ts...>> &
it_is(const std::tuple<Ts...> &t) {
  return details::_gen_ref<std_tuplelike<std::tuple<Ts...>>>();
}
template <class... Ts> struct is_std_tuplelike<std::tuple<Ts...>> : yes {};

// pair
template <class T1, class T2>
constexpr const std_tuplelike<std::pair<T1, T2>> &
it_is(const std::pair<T1, T2> &t) {
  return details::_gen_ref<std_tuplelike<std::pair<T1, T2>>>();
}
template <class T1, class T2>
struct is_std_tuplelike<std::pair<T1, T2>> : yes {};

// unregistered
template <class T> struct unregistered : proxy_base<T> {};
template <class T,
          class = std::enable_if_t<
              !std::is_convertible<const T &, const object_base<T> &>::value &&
              !std::is_scalar<T>::value && !is_std_container<T>::value &&
              !is_std_tuplelike<T>::value>>
constexpr const unregistered<T> &it_is(const T &) {
  return details::_gen_ref<unregistered<T>>();
}

// what
template <class T> constexpr decltype(auto) what(const T &v) {
  return it_is(v);
}
}