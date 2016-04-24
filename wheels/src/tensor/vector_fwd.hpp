#pragma once

#include "base_fwd.hpp"

namespace wheels {

template <class ET1, class ShapeT1, class T1, class ET2, class ShapeT2,
          class T2>
constexpr auto distance(const tensor_base<ET1, ShapeT1, T1> &t1,
                        const tensor_base<ET2, ShapeT2, T2> &t2);

template <class ET1, class ShapeT1, class T1, class ET2, class ShapeT2,
          class T2>
auto dot(const tensor_base<ET1, ShapeT1, T1> &t1,
         const tensor_base<ET2, ShapeT2, T2> &t2);

template <class E1, class ST1, class NT1, class T1, class E2, class ST2,
          class NT2, class T2>
constexpr auto cross(const tensor_base<E1, tensor_shape<ST1, NT1>, T1> &a,
                     const tensor_base<E2, tensor_shape<ST2, NT2>, T2> &b);
}