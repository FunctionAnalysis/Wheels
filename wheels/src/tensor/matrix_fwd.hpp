#pragma once

#include "base_fwd.hpp"

namespace wheels {

// matrix base
template <class T> struct matrix_base;

// auto matrix_mul(ts1, ts2);
template <class EleT, class ShapeT, class A, class B, bool AIsMat, bool BIsMat>
class matrix_mul_result;

// matrix ops
template <class E1, class ST1, class MT1, class NT1, class T1, class E2,
          class ST2, class MT2, class T2>
inline auto translate(const tensor_base<E1, tensor_shape<ST1, MT1, NT1>, T1> &m,
                      const tensor_base<E2, tensor_shape<ST2, MT2>, T2> &v);

template <class E1, class ST1, class MT1, class NT1, class T1, class E2,
          class EAngle, class ST2, class MT2, class T2>
inline auto rotate(const tensor_base<E1, tensor_shape<ST1, MT1, NT1>, T1> &m,
                   const EAngle &angle,
                   const tensor_base<E2, tensor_shape<ST2, MT2>, T2> &v);

template <class E1, class ST1, class MT1, class NT1, class T1, class E2,
          class ST2, class MT2, class T2>
inline auto scale(const tensor_base<E1, tensor_shape<ST1, MT1, NT1>, T1> &m,
                  const tensor_base<E2, tensor_shape<ST2, MT2>, T2> &v);

// camera ops
template <class E1, class ST1, class MT1, class T1, class E2, class ST2,
          class MT2, class T2, class E3, class ST3, class MT3, class T3>
inline auto
look_at_rh(const tensor_base<E1, tensor_shape<ST1, MT1>, T1> &eye,
           const tensor_base<E2, tensor_shape<ST2, MT2>, T2> &center,
           const tensor_base<E3, tensor_shape<ST3, MT3>, T3> &up);
template <class T>
inline auto perspective_fov_rh(const T &fov, const T &width, const T &height,
                               const T &z_near, const T &z_far);

template <class T>
inline auto perspective_rh(const T &fovy, const T &aspect, const T &z_near,
                           const T &z_far);
template <class T>
inline auto perspective_screen(const T &fx, const T &fy, const T &cx,
                               const T &cy, const T &z_near, const T &z_far);

template <class T>
inline auto ortho(const T &left, const T &right, const T &bottom, const T &top,
                  const T &z_near, const T &z_far);
}