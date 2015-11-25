#pragma once

#include "vector.hpp"

namespace wheels {



    // matrix transpose
    namespace tdp {
        template <class M>
        struct matrix_transpose {
            using value_type = typename std::decay_t<M>::value_type;
            M m;
            constexpr matrix_transpose(M && mm) : m(forward<M>(mm)) {}
        };

        template <class M>
        struct is_element_readable_at_subs<matrix_transpose<M>> : yes {};

        template <class M, class SubT1, class SubT2>
        constexpr decltype(auto) element_at_subs(const matrix_transpose<M> & t, 
            const SubT1 & sub1, const SubT2 & sub2) {
            return t.m.at_subs_const(sub2, sub1);
        }
    }


    // the matrix pattern
    template <class LayoutT>
    class matrix_base : public tensor_base<LayoutT> {
    public:
        constexpr auto m() const { return size(const_index<0>()); }
        constexpr auto rows() const { return m(); }
        constexpr auto n() const { return size(const_index<1>()); }
        constexpr auto cols() const { return n(); }
        constexpr auto t() const & { 
            return compose_tensor(make_shape(n(), m()), 
                tdp::matrix_transpose<const LayoutT &>(layout())); 
        }
        auto t() && {
            return compose_tensor(make_shape(n(), m()),
                tdp::matrix_transpose<LayoutT>(std::move(layout())));
        }
    };
    template <class ST, class M, class N, class DPT>
    class tensor_pattern<tensor_layout<tensor_shape<ST, M, N>, DPT>>
        : public matrix_base<tensor_layout<tensor_shape<ST, M, N>, DPT>> {};

    // is_matrix
    template <class T> struct is_matrix : no {};
    template <class ST, class M, class N, class DPT>
    struct is_matrix<tensor_layout<tensor_shape<ST, M, N>, DPT>> : yes {};


    // matrix product
    namespace tdp {
        template <class T, class A, class B,
            bool AMat = is_matrix<std::decay_t<A>>::value,
            bool BMat = is_matrix<std::decay_t<B>>::value>
        struct matrix_prod {
            using value_type = T;
            A a;
            B b;
            constexpr matrix_prod(A && aa, B && bb) : a(forward<A>(aa)), b(forward<B>(bb)) {
                assert(a.size(const_index<1>()) == b.size(const_index<0>()));
            }
        };

        template <class T, class A, class B>
        struct is_element_readable_at_subs<matrix_prod<T, A, B>> : yes {};

        template <class T, class A, class B, class SubT1, class SubT2>
        T element_at_subs(const matrix_prod<T, A, B, true, true> & prod, const SubT1 & sub1, const SubT2 & sub2) {
            T result = 0;
            for (size_t i = 0; i < prod.a.n(); i++) {
                result += prod.a.at_subs_const(sub1, i) * prod.b.at_subs_const(i, sub2);
            }
            return result;
        }
        template <class T, class A, class B, class SubT>
        T element_at_subs(const matrix_prod<T, A, B, true, false> & prod, const SubT & sub) {
            T result = 0;
            for (size_t i = 0; i < prod.a.n(); i++) {
                result += prod.a.at_subs_const(sub, i) * prod.b.at_subs_const(i);
            }
            return result;
        }

        template <class T, class A, class B>
        constexpr auto matrix_prod_to_tensor(matrix_prod<T, A, B, true, true> && prod) {
            assert(prod.a.size(const_index<1>()) == prod.b.size(const_index<0>()));
            using _shape_v_t = typename std::decay_t<A>::shape_type::value_type;
            return compose_tensor(make_shape<_shape_v_t>(prod.a.size(const_index<0>()), prod.b.size(const_index<1>())), std::move(prod));
        }

        template <class T, class A, class B>
        constexpr auto matrix_prod_to_tensor(matrix_prod<T, A, B, true, false> && prod) {
            assert(prod.a.size(const_index<1>()) == prod.b.size(const_index<0>()));
            using _shape_v_t = typename std::decay_t<A>::shape_type::value_type;
            return compose_tensor(make_shape<_shape_v_t>(prod.a.size(const_index<0>())), std::move(prod));
        }
    }

    template <class A, class B,
    class AT = std::decay_t<A>, class BT = std::decay_t<B>,
        bool Valid = is_matrix<AT>::value && (is_matrix<BT>::value || is_vector<BT>::value),
    class = std::enable_if_t<Valid >>
        constexpr auto operator * (A && a, B && b) {
        using _v_t = std::common_type_t<typename AT::value_type, typename BT::value_type>;
        return tdp::matrix_prod_to_tensor(tdp::matrix_prod<_v_t, A, B>(forward<A>(a), forward<B>(b)));
    }



    // matrix transform

    // rotate
    template <class LayoutT, class K>
    mat_<K, 3, 3> make_rotate3(const vector_base<LayoutT> & axis, const K & angle) {
        assert(axis.numel() == 3);
        auto a = axis.normalized();
        auto l = a[0], m = a[1], n = a[2];
        auto cosv = std::cos(angle);
        auto sinv = std::sin(angle);
        return mat_<K, 3, 3>(with_elements,
            l*l*(1 - cosv) + cosv, m*l*(1 - cosv) - n*sinv, n*l*(1 - cosv) + m*sinv,
            l*m*(1 - cosv) + n*sinv, m*m*(1 - cosv) + cosv, n*m*(1 - cosv) - l*sinv,
            l*n*(1 - cosv) - m*sinv, m*n*(1 - cosv) + l*sinv, n*n*(1 - cosv) + cosv);
    }

    template <class LayoutT, class K>
    mat_<K, 4, 4> make_rotate4(const vector_base<LayoutT> & axis, const K & angle) {
        assert(axis.numel() == 3);
        auto a = axis.normalized();
        auto l = a[0], m = a[1], n = a[2];
        auto cosv = std::cos(angle);
        auto sinv = std::sin(angle);
        return mat_<K, 4, 4>(with_elements,
            l*l*(1 - cosv) + cosv, m*l*(1 - cosv) - n*sinv, n*l*(1 - cosv) + m*sinv, 0,
            l*m*(1 - cosv) + n*sinv, m*m*(1 - cosv) + cosv, n*m*(1 - cosv) - l*sinv, 0,
            l*n*(1 - cosv) - m*sinv, m*n*(1 - cosv) + l*sinv, n*n*(1 - cosv) + cosv, 0,
            0, 0, 0, 1);
    }


    // camera functions with matrix
    // make a lookat view matrix
    template <class LT1, class LT2, class LT3>
    mat_<typename LT1::value_type, 4, 4> make_look_at(const vector_base<LT1> & eye,
        const vector_base<LT2> & center,
        const vector_base<LT3> & up) {
        assert(eye.numel() == 3 && center.numel() == 3 && up.numel() == 3);
        auto zaxis = (center.layout() - eye.layout()).normalized();
        auto xaxis = cross(up.layout(), zaxis).normalized();
        auto yaxis = cross(zaxis, xaxis);
       /* return mat_<typename LT1::value_type, 4, 4>(with_elements,
            -xaxis[0], yaxis[0], -zaxis[0], 0,
            -xaxis[1], yaxis[1], -zaxis[1], 0,
            -xaxis[2], yaxis[2], -zaxis[2], 0,
            dot(xaxis, eye.layout()), -dot(yaxis, eye.layout()), dot(zaxis, eye.layout()), 1).t();*/
        return mat_<typename LT1::value_type, 4, 4>(with_elements,
            -xaxis[0], -xaxis[1], -xaxis[2],  dot(xaxis, eye.layout()),
             yaxis[0],  yaxis[1],  yaxis[2], -dot(yaxis, eye.layout()),
            -zaxis[0], -zaxis[1], -zaxis[2],  dot(zaxis, eye.layout()),
                    0,         0,         0,                        1);
    }



    // make a perspective projection matrix
    template <class T>
    mat_<T, 4, 4> make_perspective(const T & fovyRadians, const T & aspect,
        const T & nearZ, const T & farZ) {
        T cotan = T(1.0) / std::tan(fovyRadians / 2.0);
        //return mat_<T, 4, 4>(with_elements,
        //    cotan / aspect, 0, 0, 0,
        //    0,          cotan, 0, 0,
        //    0,              0, (farZ + nearZ) / (nearZ - farZ), -1,
        //    0,              0, (2 * farZ * nearZ) / (nearZ - farZ), 0).t();
        return mat_<T, 4, 4>(with_elements,
            cotan / aspect, 0, 0,                                0,
            0,          cotan, 0,                                0,
            0,              0, (farZ + nearZ) / (nearZ - farZ), (2 * farZ * nearZ) / (nearZ - farZ),
            0,              0, -1,                               0);
    }
    template <class T>
    mat_<T, 4, 4> make_perspective(const T & fx, const T & fy,
        const T & cx, const T & cy,
        const T & nearZ, const T & farZ) {
        /*return mat_<T, 4, 4>(with_elements,
            fx / cx, 0, 0, 0,
            0, fy / cy, 0, 0,
            0, 0, (farZ + nearZ) / (nearZ - farZ), -1,
            0, 0, (2 * farZ * nearZ) / (nearZ - farZ), 0).t();*/
        return mat_<T, 4, 4>(with_elements,
            fx / cx, 0, 0, 0,
            0, fy / cy, 0, 0,
            0, 0, (farZ + nearZ) / (nearZ - farZ), (2 * farZ * nearZ) / (nearZ - farZ),
            0, 0,      -1, 0);
    }


}
