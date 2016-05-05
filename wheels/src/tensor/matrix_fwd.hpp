#pragma once

#include "base_fwd.hpp"

namespace wheels {

// matrix base
template <class T> struct matrix_base;

// auto matrix_mul(ts1, ts2);
template <class EleT, class ShapeT, class A, class B, bool AIsMat, bool BIsMat>
class matrix_mul_result;

template <class E1, class ST1, class MT1, class NT1, class T1, class E2,
          class ST2, class MT2, class T2>
inline auto translate(const tensor_base<E1, tensor_shape<ST1, MT1, NT1>, T1> &m,
                      const tensor_base<E2, tensor_shape<ST2, MT2>, T2> &v);

template <class E1, class ST1, class MT1, class NT1, class T1, class E2,
          class EAngle, class ST2, class MT2, class T2>
inline auto rotate(const tensor_base<E1, tensor_shape<ST1, MT1, NT1>, T1> &m,
                   const EAngle &angle,
                   const tensor_base<E2, tensor_shape<ST2, MT2>, T2> &v);
}