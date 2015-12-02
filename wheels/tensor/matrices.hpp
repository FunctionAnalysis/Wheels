#pragma once

#include "categories.hpp"
#include "functions.hpp"

namespace wheels {


    // vector
    template <class CategoryT, bool Writable> class vector;
    template <class CategoryT, class ST, class SizeT>
    class tensor_specific_shape<CategoryT, tensor_shape<ST, SizeT>>
        : public vector<CategoryT, tensor_traits::writable<CategoryT>::value> {};


    template <class CategoryT, bool Writable>
    class vector : public tensor_specific_shape_base<CategoryT> {
    public:
        constexpr decltype(auto) x() const { return at_index_const(0); }
        constexpr decltype(auto) y() const { return at_index_const(1); }
        constexpr decltype(auto) z() const { return at_index_const(2); }
        constexpr decltype(auto) w() const { return at_index_const(3); }
        constexpr decltype(auto) red() const { return at_index_const(0); }
        constexpr decltype(auto) green() const { return at_index_const(1); }
        constexpr decltype(auto) blue() const { return at_index_const(2); }
        constexpr decltype(auto) alpha() const { return at_index_const(3); }
        constexpr auto normalized() const { return category() / norm(); }
    };
    template <class CategoryT>
    class vector<CategoryT, true> : public tensor_specific_shape_base<CategoryT> {
    public:
        constexpr decltype(auto) x() const { return at_index_const(0); }
        constexpr decltype(auto) y() const { return at_index_const(1); }
        constexpr decltype(auto) z() const { return at_index_const(2); }
        constexpr decltype(auto) w() const { return at_index_const(3); }
        constexpr decltype(auto) red() const { return at_index_const(0); }
        constexpr decltype(auto) green() const { return at_index_const(1); }
        constexpr decltype(auto) blue() const { return at_index_const(2); }
        constexpr decltype(auto) alpha() const { return at_index_const(3); }
        decltype(auto) x() { return at_index_nonconst(0); }
        decltype(auto) y() { return at_index_nonconst(1); }
        decltype(auto) z() { return at_index_nonconst(2); }
        decltype(auto) w() { return at_index_nonconst(3); }
        decltype(auto) red() { return at_index_nonconst(0); }
        decltype(auto) green() { return at_index_nonconst(1); }
        decltype(auto) blue() { return at_index_nonconst(2); }
        decltype(auto) alpha() { return at_index_nonconst(3); }
        constexpr auto normalized() const { return category() / norm(); }
    };


    // matrix
    template <class CategoryT, bool Writable> class matrix;
    template <class CategoryT, class ST, class MT, class NT>
    class tensor_specific_shape<CategoryT, tensor_shape<ST, MT, NT>>
        : public matrix<CategoryT, tensor_traits::writable<CategoryT>::value> {};

    template <class CategoryT, bool Writable>
    class matrix : public tensor_specific_shape_base<CategoryT> {
    public:
        constexpr auto rows() const { return size(const_index<0>()); }
        constexpr auto cols() const { return size(const_index<1>()); }
        constexpr auto t() const & { return permute(category(), const_index<1>(), const_index<0>()); }
        constexpr auto t() const && { return permute(std::move(category()), const_index<1>(), const_index<0>()); }
    };



    // print
    template <class CategoryT, bool Writable>
    inline std::ostream & operator << (std::ostream & os, const vector<CategoryT, Writable> & v) {
        os << "[";
        for (size_t ind = 0; ind < v.numel() - 1; ind++) {
            os << v[ind] << ", ";
        }
        os << v[index_tags::last] << "]";
        return os;
    }

    // print
    template <class CategoryT, bool Writable>
    inline std::ostream & operator << (std::ostream & os, const matrix<CategoryT, Writable> & m) {
        for (size_t r = 0; r < m.rows(); r++) {
            os << "[";
            for (size_t c = 0; c < m.cols() - 1; c++) {
                os << m(r, c) << ", ";
            }
            os << m(r, index_tags::last) << "]\n";
        }
        return os;
    }


    // cross
    template <class CategoryT1, class CategoryT2, bool W1, bool W2>
    constexpr auto cross(const vector<CategoryT1, W1> & a, const vector<CategoryT2, W2> & b) {
        using value_t = std::common_type_t<typename CategoryT1::value_type, typename CategoryT2::value_type>
            assert(a.numel() == 3 && b.numel() == 3);
        return vec_<value_t, 3>(with_elements,
            a.y() * b.z() - a.z() * b.y(),
            a.z() * b.x() - a.x() * b.z(),
            a.x() * b.y() - a.y() * b.x());
    }


    // matrix mul
    template <class T, class A, class B, bool AIsMat, bool BIsMat> class matrix_mul_result;
    namespace tensor_traits {
        template <class ShapeT, class T, class A, class B, bool AIsMat, bool BIsMat>
        struct readable_at_subs<tensor_category<ShapeT, matrix_mul_result<T, A, B, AIsMat, BIsMat>>> : yes {};
        template <class ShapeT, class T, class A, class B, bool AIsMat, bool BIsMat, class ... SubTs>
        constexpr T at_subs_const_impl(
            const tensor_category<ShapeT, matrix_mul_result<T, A, B, AIsMat, BIsMat>> & mul,
            const SubTs & ... subs) {
            return mul.data_provider().at_subs(subs ...);
        }
    }

    // matrix + matrix -> matrix
    template <class T, class A, class B>
    class matrix_mul_result<T, A, B, true, true> {
    public:
        using value_type = T;
        constexpr matrix_mul_result(A && aa, B && bb)
            : _a(forward<A>(aa)), _b(forward<B>(bb)) {}        
        template <class SubT1, class SubT2>
        decltype(auto) at_subs(const SubT1 & s1, const SubT2 & s2) const {
            T result = types<T>::zero();
            for (size_t i = 0; i < _a.size(const_index<1>()); i++) {
                result += _a.at_subs_const(s1, i) * _b.at_subs_const(i, s2);
            }
            return result;
        }
    private:
        A _a;
        B _b;
    };

    template <class ST1, class MT1, class NT1, class DPT1, 
        class ST2, class MT2, class NT2, class DPT2>
    struct overloaded<binary_op_mul,
        tensor_category<tensor_shape<ST1, MT1, NT1>, DPT1>,
        tensor_category<tensor_shape<ST2, MT2, NT2>, DPT2>> {        
        constexpr overloaded() {}

        template <class A, class B>
        constexpr auto operator()(A && a, B && b) const {
            assert(a.size(const_index<1>()) == b.size(const_index<0>()));
            using shape_v_t = std::common_type_t<
                typename std::decay_t<A>::shape_type::value_type, 
                typename std::decay_t<B>::shape_type::value_type
            >;
            using result_v_t = std::common_type_t<
                typename std::decay_t<A>::value_type, 
                typename std::decay_t<B>::value_type
            >;
            return make_tensor(make_shape<shape_v_t>(a.size(const_index<0>()), b.size(const_index<1>())),
                matrix_mul_result<result_v_t, A, B, true, true>(forward<A>(a), forward<B>(b)));
        }
    };

    // matrix + vector -> vector
    template <class T, class A, class B>
    class matrix_mul_result<T, A, B, true, false> {
    public:
        using value_type = T;
        constexpr matrix_mul_result(A && aa, B && bb)
            : _a(forward<A>(aa)), _b(forward<B>(bb)) {}
        template <class SubT>
        decltype(auto) at_subs(const SubT & s) const {
            T result = types<T>::zero();
            for (size_t i = 0; i < _a.size(const_index<1>()); i++) {
                result += _a.at_subs_const(s, i) * _b.at_subs_const(i);
            }
            return result;
        }
    private:
        A _a;
        B _b;
    };

    template <class ST1, class MT1, class NT1, class DPT1,
        class ST2, class MT2, class DPT2>
    struct overloaded<binary_op_mul,
        tensor_category<tensor_shape<ST1, MT1, NT1>, DPT1>,
        tensor_category<tensor_shape<ST2, MT2>, DPT2>> {
        constexpr overloaded() {}

        template <class A, class B>
        constexpr auto operator()(A && a, B && b) const {
            assert(a.size(const_index<1>()) == b.size(const_index<0>()));
            using shape_v_t = std::common_type_t<
                typename std::decay_t<A>::shape_type::value_type,
                typename std::decay_t<B>::shape_type::value_type
            >;
            using result_v_t = std::common_type_t<
                typename std::decay_t<A>::value_type,
                typename std::decay_t<B>::value_type
            >;
            return make_tensor(make_shape<shape_v_t>(a.size(const_index<0>())),
                matrix_mul_result<result_v_t, A, B, true, false>(forward<A>(a), forward<B>(b)));
        }
    };

    // vector + matrix -> vector
    template <class T, class A, class B>
    class matrix_mul_result<T, A, B, false, true> {
    public:
        using value_type = T;
        constexpr matrix_mul_result(A && aa, B && bb)
            : _a(forward<A>(aa)), _b(forward<B>(bb)) {}
        template <class SubT>
        decltype(auto) at_subs(const SubT & s) const {
            T result = types<T>::zero();
            for (size_t i = 0; i < _a.size(const_index<0>()); i++) {
                result += _a.at_subs_const(i) * _b.at_subs_const(i, s);
            }
            return result;
        }
    private:
        A _a;
        B _b;
    };

    template <class ST1, class MT1, class DPT1,
        class ST2, class MT2, class NT2, class DPT2>
    struct overloaded<binary_op_mul,
        tensor_category<tensor_shape<ST1, MT1>, DPT1>,
        tensor_category<tensor_shape<ST2, MT2, NT2>, DPT2>> {

        constexpr overloaded() {}

        template <class A, class B>
        constexpr auto operator()(A && a, B && b) const {
            assert(a.size(const_index<0>()) == b.size(const_index<0>()));
            using shape_v_t = std::common_type_t<
                typename std::decay_t<A>::shape_type::value_type,
                typename std::decay_t<B>::shape_type::value_type
            >;
            using result_v_t = std::common_type_t<
                typename std::decay_t<A>::value_type,
                typename std::decay_t<B>::value_type
            >;
            return make_tensor(make_shape<shape_v_t>(b.size(const_index<1>())),
                matrix_mul_result<result_v_t, A, B, false, true>(forward<A>(a), forward<B>(b)));
        }
    };






}


