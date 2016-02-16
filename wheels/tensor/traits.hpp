#pragma once

#include <type_traits>
#include "../core/utility.hpp"

namespace wheels {

// tensor_element_types
template <class T> struct tensor_element_types {
  using storable = T;
  using ref = T &; // x = ...
  using const_ref = T const &;
  using pointer = T *;
  using const_pointer = T const *;
};

template <class T> struct tensor_element_types<const T> {
  using storable = T;
  using ref = T const &; // x = ...
  using const_ref = T const &;
  using pointer = T const *;
  using const_pointer = T const *;
};

template <class T> struct tensor_element_types<T &> {
  static_assert(always<bool, false, T>::value, "reference type not supported");
};
}