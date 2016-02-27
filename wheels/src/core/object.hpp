#pragma once

#include <type_traits>

namespace wheels {
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
template <class T> struct other {};

template <class T, class = std::enable_if_t<!std::is_convertible<
                       const T &, const object<T> &>::value>>
constexpr const other<T> &identify(const T &t) {
  return *static_cast<const other<T> *>(nullptr);
}
}