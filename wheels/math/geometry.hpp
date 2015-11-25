#pragma once

#include "tensor.hpp"
#include "tensor_op.hpp"

namespace wheels {

    // tensor extensions

    // vectors
    namespace tdp {
        template <class T, size_t Idx>
        struct unit_at {
            using value_type = T;
            constexpr unit_at() {}
            template <class Archiver>
            void serialize(Archiver &) {}
        };
        // accessing elements
        template <class T, size_t Idx> struct is_element_readable_at_index<unit_at<T, Idx>> : yes {};
        template <class T, size_t Idx, class IndexT>
        constexpr T element_at_index(const unit_at<T, Idx> & a, const IndexT & index) {
            return index == Idx ? 1 : 0;
        }
    }

    // unit_x/y/z
    template <class T = double, class ST = size_t, ST Size = 3>
    constexpr auto unit_x(const const_ints<ST, Size> & s = const_ints<ST, Size>()) { 
        static_assert(Size > 0, "vector size too small");
        return compose_tensor(make_shape<ST>(s), tdp::unit_at<T, 0>()); 
    }
    template <class T = double, class ST = size_t, ST Size = 3>
    constexpr auto unit_y(const const_ints<ST, Size> & s = const_ints<ST, Size>()) {
        static_assert(Size > 1, "vector size too small");
        return compose_tensor(make_shape<ST>(s), tdp::unit_at<T, 1>());
    }
    template <class T = double, class ST = size_t, ST Size = 3>
    constexpr auto unit_z(const const_ints<ST, Size> & s = const_ints<ST, Size>()) {
        static_assert(Size > 2, "vector size too small");
        return compose_tensor(make_shape<ST>(s), tdp::unit_at<T, 2>());
    }

    // dot


    // cross


    template <class LayoutT>
    class vector_methods : public tensor_base<LayoutT> {
    public:
        auto length() const { return norm(); }
        auto normalized() const & { return layout() / length(); }
        auto normalized() && {return std::move(layout()) / length(); }

        constexpr auto n() const { return size(const_index<0>()); }
        
        constexpr decltype(auto) x() const { return at_index_const(0); }
        constexpr decltype(auto) y() const { return at_index_const(1); }
        constexpr decltype(auto) z() const { return at_index_const(2); }
        constexpr decltype(auto) w() const { return at_index_const(3); }

        decltype(auto) x() { return at_index_nonconst(0); }
        decltype(auto) y() { return at_index_nonconst(1); }
        decltype(auto) z() { return at_index_nonconst(2); }
        decltype(auto) w() { return at_index_nonconst(3); }

        constexpr decltype(auto) red() const { return at_index_const(0); }
        constexpr decltype(auto) green() const { return at_index_const(1); }
        constexpr decltype(auto) blue() const { return at_index_const(2); }
        constexpr decltype(auto) alpha() const { return at_index_const(3); }

        decltype(auto) red() { return at_index_nonconst(0); }
        decltype(auto) green() { return at_index_nonconst(1); }
        decltype(auto) blue() { return at_index_nonconst(2); }
        decltype(auto) alpha() { return at_index_nonconst(3); }
    };
    template <class ST, class SizeT, class DPT>
    class tensor_extended<tensor_layout<tensor_shape<ST, SizeT>, DPT>>
        : public vector_methods<tensor_layout<tensor_shape<ST, SizeT>, DPT>> {
    };

    // is_vector
    template <class T> struct is_vector : no {};
    template <class ST, class SizeT, class DPT>
    struct is_vector<tensor_layout<tensor_shape<ST, SizeT>, DPT>> : yes {};







    // matrices
    template <class LayoutT>
    class matrix_methods : public tensor_base<LayoutT> {
    public:
        constexpr auto m() const { return size(const_index<0>()); }
        constexpr auto rows() const { return m(); }
        constexpr auto n() const { return size(const_index<1>()); }
        constexpr auto cols() const { return n(); }
    };
    template <class ST, class M, class N, class DPT>
    class tensor_extended<tensor_layout<tensor_shape<ST, M, N>, DPT>>
        : public matrix_methods<tensor_layout<tensor_shape<ST, M, N>, DPT>> {
    };

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
        class = std::enable_if_t<Valid>>
    constexpr auto operator * (A && a, B && b) {        
        using _v_t = std::common_type_t<typename AT::value_type, typename BT::value_type>;
        return tdp::matrix_prod_to_tensor(tdp::matrix_prod<_v_t, A, B>(forward<A>(a), forward<B>(b)));
    }



    // matrix transform

    // rotate
    template <class ST, class SizeT, class DPT, class K>
    mat_<K, 3, 3> make_rotate3(const tensor_layout<tensor_shape<ST, SizeT>, DPT> & axis, const K & angle) {
        assert(axis.numel() == 3);
        auto a = axis.normalized();
        auto l = a[0], m = a[1], n = a[2];
        auto cosv = std::cos(angle);
        auto sinv = std::sin(angle);
        return mat_<K, 3, 3>(with_elements,
            l*l*(1 - cosv) + cosv, m*l*(1 - cosv) - n*sinv, n*l*(1 - cosv) + m*sinv,
            l*m*(1 - cosv) + n*sinv, m*m*(1 - cosv) + cosv, n*m*(1 - cosv) - l*sinv,
            l*n*(1 - cosv) - m*sinv, m*n*(1 - cosv) + l*sinv, n*n*(1 - cosv) + cosv);
        /*return mat_<K, 3, 3>(with_elements,
            l*l*(1 - cosv) + cosv, m*l*(1 - cosv) + n*sinv, n*l*(1 - cosv) - m*sinv,
            l*m*(1 - cosv) - n*sinv, m*m*(1 - cosv) + cosv, n*m*(1 - cosv) + l*sinv,
            l*n*(1 - cosv) + m*sinv, m*n*(1 - cosv) - l*sinv, n*n*(1 - cosv) + cosv);*/
    }

    template <class ST, class SizeT, class DPT, class K>
    mat_<K, 4, 4> make_rotate4(const tensor_layout<tensor_shape<ST, SizeT>, DPT> & axis, const K & angle) {
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

    // translate


}