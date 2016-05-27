#pragma once

#include "tensor_base_fwd.hpp"

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

template <class E1, class ST1, class MT1, class T1, class E2, class ST2,
          class MT2, class T2, class E3, class ST3, class MT3, class T3>
inline auto
look_at_rh(const tensor_base<E1, tensor_shape<ST1, MT1>, T1> &eye,
           const tensor_base<E2, tensor_shape<ST2, MT2>, T2> &center,
           const tensor_base<E3, tensor_shape<ST3, MT3>, T3> &up);
}