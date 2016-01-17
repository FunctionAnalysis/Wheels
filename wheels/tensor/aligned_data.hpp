#pragma once

#include "base.hpp"

namespace wheels {

// tensor_with_aligned_data
template <class ShapeT, class ET, class StepsT, bool StaticSteps, class T>
class tensor_with_aligned_data;

template <class ShapeT, class ET, class StepsT, class T>
class tensor_with_aligned_data<ShapeT, ET, StepsT, true, T>
    : public tensor_base<ShapeT, ET, T> {
  static_assert(is_combination<StepsT>::value,
                "StepsT should be a combination<...>");

public:
  explicit tensor_with_aligned_data(const StepsT &) {}

public:
  constexpr StepsT steps() const { return StepsT(); }
  constexpr decltype(auto) ptr() const { return ::wheels::ptr_of(derived()); }
  decltype(auto) ptr() { return ::wheels::ptr_of(derived()); }
};

template <class ShapeT, class ET, class StepsT, class T>
class tensor_with_aligned_data<ShapeT, ET, StepsT, false, T>
    : public tensor_base<ShapeT, ET, T> {
  static_assert(is_combination<StepsT>::value,
                "StepsT should be a combination<...>");

public:
  explicit tensor_with_aligned_data(const StepsT &s) : _steps(s) {}

public:
  constexpr const StepsT &steps() const { return _steps; }
  constexpr decltype(auto) ptr() const { return ::wheels::ptr_of(derived()); }
  decltype(auto) ptr() { return ::wheels::ptr_of(derived()); }

private:
  StepsT _steps;
};



}
