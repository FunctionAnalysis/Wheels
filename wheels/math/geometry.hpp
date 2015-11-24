#pragma once

#include "tensor.hpp"
#include "tensor_op.hpp"

namespace wheels {

    template <class T, size_t N>
    using point_ = vec_<T, N>;

    template <class T, class = std::enable_if<is_tensor_layout<std::decay_t<T>>::value>>
    constexpr auto normalize(T && t) {
        return forward<T>(t) / t.norm();
    }


    // matrix transform
    template <class T, class K>
    mat_<T, 3, 3> make_rotate3(const vec_<T, 3> & axis, const K & angle) {
        auto a = normalize(axis);
        double l = a[0], m = a[1], n = a[2];
        double cosv = std::cos(angle), sinv = std::sin(angle);
        return mat_<T, 3, 3>(with_elements,
            l*l*(1 - cosv) + cosv, m*l*(1 - cosv) - n*sinv, n*l*(1 - cosv) + m*sinv,
            l*m*(1 - cosv) + n*sinv, m*m*(1 - cosv) + cosv, n*m*(1 - cosv) - l*sinv,
            l*n*(1 - cosv) - m*sinv, m*n*(1 - cosv) + l*sinv, n*n*(1 - cosv) + cosv);
    }





}