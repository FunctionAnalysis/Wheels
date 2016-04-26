#pragma once

#include "../core/const_ints.hpp"

#include "base_fwd.hpp"

namespace wheels {

// cat_result
template <class ET, class ShapeT, size_t Axis, class T1, class T2>
class cat_result;

namespace details {
template <size_t Axis, class ShapeT1, class ET1, class T1, class ShapeT2,
          class ET2, class T2, class TT1, class TT2>
constexpr auto _cat_tensor_at(const const_index<Axis> &axis,
                              const tensor_base<ET1, ShapeT1, T1> &,
                              const tensor_base<ET2, ShapeT2, T2> &, TT1 &&in1,
                              TT2 &&in2);
}

template <size_t Axis, class T1, class T2>
constexpr auto cat_at(const const_index<Axis> &axis, T1 &&in1, T2 &&in2)
    -> decltype(details::_cat_tensor_at(axis, in1, in2, std::forward<T1>(in1),
                                        std::forward<T2>(in2))) {
  return details::_cat_tensor_at(axis, in1, in2, std::forward<T1>(in1),
                                 std::forward<T2>(in2));
}

// cat2 (cat_at 0)
template <class T1, class T2>
constexpr auto cat2(T1 &&in1, T2 &&in2)
    -> decltype(cat_at(const_index<0>(), std::forward<T1>(in1),
                       std::forward<T2>(in2))) {
  return cat_at(const_index<0>(), std::forward<T1>(in1), std::forward<T2>(in2));
}
}
