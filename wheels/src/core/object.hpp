#pragma once

#include <type_traits>

namespace wheels {

// object
template <class T> struct object {
  constexpr const T &derived() const & { return static_cast<const T &>(*this); }
  T &derived() & { return static_cast<T &>(*this); }
  T &&derived() && { return static_cast<T &&>(*this); }
};
}