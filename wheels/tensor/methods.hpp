#pragma once

#include "tensor.hpp"

namespace wheels {

// distance
template <class ShapeT1, class ET1, class T1, class ShapeT2, class ET2,
          class T2>
constexpr auto distance(const tensor_base<ShapeT1, ET1, T1> &t1,
                        const tensor_base<ShapeT2, ET2, T2> &t2) {
  return norm(t1.derived() - t2.derived());
}

// Scalar dot_product(ts1, ts2);
template <class ShapeT1, class ET1, class T1, class ShapeT2, class ET2,
          class T2>
auto dot(const tensor_base<ShapeT1, ET1, T1> &t1,
         const tensor_base<ShapeT2, ET2, T2> &t2) {
  using result_t = std::common_type_t<ET1, ET2>;
  assert(shape_of(t1.derived()) == shape_of(t2.derived()));
  result_t result = 0.0;
  for_each_element([&result](auto &&e1, auto &&e2) { result += e1 * e2; },
                   t1.derived(), t2.derived());
  return result;
}

// auto cross(ts1, ts2);
template <class ST1, class NT1, class E1, class T1, class ST2, class NT2,
          class E2, class T2>
constexpr auto cross(const tensor_base<tensor_shape<ST1, NT1>, E1, T1> &a,
                     const tensor_base<tensor_shape<ST2, NT2>, E2, T2> &b) {
  using result_t = std::common_type_t<E1, E2>;
  return vec_<result_t, 3>(a.y() * b.z() - a.z() * b.y(),
                           a.z() * b.x() - a.x() * b.z(),
                           a.x() * b.y() - a.y() * b.x());
}

// constants
// template <>

// eye

// iota

// meshgrid
}