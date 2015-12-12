#pragma once

#include "base.hpp"

namespace wheels {

template <class ShapeT, class ET, class AT, class BT>
class solve_result
    : public tensor_base<ShapeT, ET, solve_result<ShapeT, ET, AT, BT>> {};

// min ||a*x-b||
template <class MT1, class NT1, class ET1, class T1, class MT2, class ET2,
          class T2, class MT3, class ET3, class T3>
auto solve(const tensor_base<tensor_shape<size_t, MT1, NT1>, ET1, T1> &a,
           const tensor_base<tensor_shape<size_t, MT2>, ET2, T2> &b) {
  assert(a.size(const_index<0>()) == b.size(const_index<0>()));
#ifdef wheels_with_opencv
  // todo
#endif
}
}