#pragma once

#include <iostream>

#include "../core/operators.hpp"

#include "tensor.hpp"

namespace wheels {


    // constants
    namespace ts_traits {
        // readable
        template <class ShapeT, class E>
        struct readable_at_index<ts_category<ShapeT, constant<E>>> : yes {};

        template <class ShapeT, class E, class IndexT>
        constexpr const E & at_index_const_impl(const ts_category<ShapeT, constant<E>> & a, const IndexT & ind) {
            return a.data_provider().val;
        }

        template <class ShapeT, class E>
        struct readable_at_subs<ts_category<ShapeT, constant<E>>> : yes {};

        template <class ShapeT, class E, class ... SubTs>
        constexpr const E & at_subs_const_impl(const ts_category<ShapeT, constant<E>> & a, const SubTs & ... subs) {
            return a.data_provider().val;
        }
    }

    template <class T, class ST, class ... SizeTs>
    constexpr auto constants(const tensor_shape<ST, SizeTs ...> & shape, T && val) {
        return compose_category(shape, constant<std::decay_t<T>>(forward<T>(val)));
    }

    // zeros
    template <class T = double, class ST, class ... SizeTs>
    constexpr auto zeros(const tensor_shape<ST, SizeTs ...> & shape) {
        return compose_category(shape, constant<T>(0));
    }
    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto zeros(const SizeTs & ... sizes) {
        return compose_category(make_shape<ST>(sizes...), constant<T>(types<T>::zero()));
    }

    // ones
    template <class T = double, class ST, class ... SizeTs>
    constexpr auto ones(const tensor_shape<ST, SizeTs ...> & shape) {
        return compose_category(shape, constant<T>(1));
    }
    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto ones(const SizeTs & ... sizes) {
        return compose_category(make_shape<ST>(sizes...), constant<T>(1));
    }



    // meshgrid
    template <class T, size_t Idx>
    struct subs_component {
        using value_type = T;
        constexpr subs_component(){}
        template <class Archive>
        void serialize(Archive & ar) {}
    };

    namespace ts_traits {
        template <class ShapeT, class T, size_t Idx>
        struct readable_at_subs<ts_category<ShapeT, subs_component<T, Idx>>> : yes {};

        template <class ShapeT, class T, size_t Idx, class ... SubTs>
        constexpr T at_subs_const_impl(
            const ts_category<ShapeT, subs_component<T, Idx>> & a,
            const SubTs & ... subs) {
            return (T)std::get<Idx>(std::forward_as_tuple(subs ...));
        }
    }

    namespace details {
        template <class T, class ShapeT, size_t ... Is>
        constexpr auto _meshgrid_seq(const ShapeT & shape, const_ints<size_t, Is...>) {
            return std::make_tuple(compose_category(shape, subs_component<T, Is>())...);
        }
    }

    template <class T = double, class ST, class ... SizeTs>
    constexpr auto meshgrid(const tensor_shape<ST, SizeTs ...> & shape) {
        return details::_meshgrid_seq<T>(shape, make_const_sequence(const_size<sizeof...(SizeTs)>()));
    }
    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto meshgrid(const SizeTs & ... sizes) {
        return details::_meshgrid_seq<T>(make_shape(sizes ...), make_const_sequence(const_size<sizeof...(SizeTs)>()));
    }




    // unit axis vector
    template <class T, size_t Idx>
    struct unit_at {
        using value_type = T;
        constexpr unit_at() {}
        template <class Archiver>
        void serialize(Archiver &) {}
    };

    namespace ts_traits {
        template <class ShapeT, class T, size_t Idx>
        struct readable_at_index<ts_category<ShapeT, unit_at<T, Idx>>> : yes {};

        template <class ShapeT, class T, size_t Idx, class IndexT>
        constexpr T at_index_const_impl(const ts_category<ShapeT, unit_at<T, Idx>> & a, const IndexT & index) {
            return index == Idx ? 1 : 0;
        }
    }

    // unit_x/y/z
    template <class T = double, class ST = size_t, class SizeT = const_ints<ST, 3>>
    constexpr auto unit_x(const SizeT & s = SizeT()) {
        return compose_category(make_shape<ST>(s), unit_at<T, 0>());
    }
    template <class T = double, class ST = size_t, class SizeT = const_ints<ST, 3>>
    constexpr auto unit_y(const SizeT & s = SizeT()) {
        return compose_category(make_shape<ST>(s), unit_at<T, 1>());
    }
    template <class T = double, class ST = size_t, class SizeT = const_ints<ST, 3>>
    constexpr auto unit_z(const SizeT & s = SizeT()) {
        return compose_category(make_shape<ST>(s), unit_at<T, 2>());
    }





    
    // ewise op
    template <class T, class OpT, class ... InputTs>
    struct ewise_op_result {
        using value_type = T;
        using op_type = OpT;
        OpT op;
        std::tuple<InputTs ...> inputs;
        
        template <class OpTT, class ... InputTTs>
        constexpr explicit ewise_op_result(OpTT && o, InputTTs && ... ins)
            : op(forward<OpTT>(o)), inputs(std::forward<InputTTs>(ins) ...) {}
        
        template <class Archive> 
        void serialize(Archive & ar) { 
            ar(op, inputs); 
        }
    };

    template <class OpT, class ... InputTs>
    constexpr auto compose_ewise_op_result(OpT && op, InputTs && ... inputs) {
        using result_t = std::decay_t<decltype(op(inputs.at_index_const(0) ...))>;
        return ewise_op_result<result_t, std::decay_t<OpT>, InputTs...>(forward<OpT>(op), 
            forward<InputTs>(inputs) ...);
    }

    namespace ts_traits {
       
        // readable
        template <class ShapeT, class T, class OpT, class ... InputTs>
        struct readable_at_index<ts_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>>> : yes {};

        template <class ShapeT, class T, class OpT, class IndexT, class ... InputTs, size_t ... Is>
        constexpr decltype(auto) _ewise_op_result_at_index_const_impl_seq(
            const ts_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>> & a,
            const IndexT & ind, const const_ints<size_t, Is...> &) {
            return a.data_provider().op(std::get<Is>(a.data_provider().inputs).at_index_const(ind) ...);
        }
        template <class ShapeT, class T, class OpT, class IndexT, class ... InputTs>
        constexpr decltype(auto) at_index_const_impl(
            const ts_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>> & a, 
            const IndexT & ind) {
            return _ewise_op_result_at_index_const_impl_seq(a, ind, 
                make_const_sequence(const_size<sizeof...(InputTs)>()));
        }

        template <class ShapeT, class T, class OpT, class ... InputTs>
        struct readable_at_subs<ts_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>>> : yes {};

        template <class ShapeT, class T, class OpT, class ... InputTs, size_t ... Is, class ... SubTs>
        constexpr decltype(auto) _ewise_op_result_at_subs_const_impl_seq(
            const ts_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>> & a,
            const const_ints<size_t, Is...> &, const SubTs & ... subs) {
            return a.data_provider().op(std::get<Is>(a.data_provider().inputs).at_subs_const(subs ...) ...);
        }
        template <class ShapeT, class T, class OpT, class ... InputTs, class ... SubTs>
        constexpr decltype(auto) at_subs_const_impl(
            const ts_category<ShapeT, ewise_op_result<T, OpT, InputTs ...>> & a,
            const SubTs & ... subs) {
            return _ewise_op_result_at_subs_const_impl_seq(a, 
                make_const_sequence(const_size<sizeof...(InputTs)>()), subs ...);
        }
    }


#define WHEELS_TS_OVERLOAD_EWISE_UNARY_OP(op, name) \
    template <class A, \
        class = std::enable_if_t<is_tensor<std::decay_t<A>>::value>, \
        wheels_distinguish_1>  \
    constexpr auto operator op (A && a) { \
        return compose_category(a.shape(), compose_ewise_op_result(unary_op_##name(), forward<A>(a))); \
    }

    WHEELS_TS_OVERLOAD_EWISE_UNARY_OP(+, plus)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_OP(-, minus)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_OP(!, not)


#define WHEELS_TS_OVERLOAD_EWISE_BINARY_OP(op, name) \
    template <class A, class B, \
        class = std::enable_if_t<is_tensor<std::decay_t<A>>::value && is_tensor<std::decay_t<B>>::value>, \
        wheels_distinguish_3> \
    constexpr auto operator op (A && a, B && b) { \
        assert(a.shape() == b.shape()); \
        return compose_category(a.shape(),  \
            compose_ewise_op_result(binary_op_##name(), forward<A>(a), forward<B>(b))); \
    }

   WHEELS_TS_OVERLOAD_EWISE_BINARY_OP(+, plus)
   WHEELS_TS_OVERLOAD_EWISE_BINARY_OP(-, minus)
           
   WHEELS_TS_OVERLOAD_EWISE_BINARY_OP(== , eq)
   WHEELS_TS_OVERLOAD_EWISE_BINARY_OP(!= , neq)
   WHEELS_TS_OVERLOAD_EWISE_BINARY_OP(<, lt)
   WHEELS_TS_OVERLOAD_EWISE_BINARY_OP(<= , lte)
   WHEELS_TS_OVERLOAD_EWISE_BINARY_OP(>, gt)
   WHEELS_TS_OVERLOAD_EWISE_BINARY_OP(>= , gte)
           
   WHEELS_TS_OVERLOAD_EWISE_BINARY_OP(&&, and)
   WHEELS_TS_OVERLOAD_EWISE_BINARY_OP(|| , or )
   WHEELS_TS_OVERLOAD_EWISE_BINARY_OP(&, bitwise_and)
   WHEELS_TS_OVERLOAD_EWISE_BINARY_OP(| , bitwise_or)



#define WHEELS_TS_OVERLOAD_EWISE_UNARY_OP_WITH_SCALAR(op, name) \
    template <class A, class B, class = \
        std::enable_if_t<\
         is_tensor<std::decay_t<A>>::value &&  \
        !is_tensor<std::decay_t<B>>::value && \
        !is_const_expr<std::decay_t<B>>::value>, wheels_distinguish_4> \
    constexpr auto operator op (A && a, B && b) { \
        return compose_category(a.shape(), \
            compose_ewise_op_result(unary_op_##name##_with<std::decay_t<B>>(forward<B>(b)), forward<A>(a))); \
    } \
    template <class A, class B, class = \
        std::enable_if_t<\
         is_tensor<std::decay_t<B>>::value && \
        !is_tensor<std::decay_t<A>>::value && \
        !is_const_expr<std::decay_t<A>>::value>, wheels_distinguish_5> \
        constexpr auto operator op (A && a, B && b) { \
        return compose_category(b.shape(), \
            compose_ewise_op_result(unary_op_##name##_with_<std::decay_t<A>>(forward<A>(a)), forward<B>(b))); \
    }


    WHEELS_TS_OVERLOAD_EWISE_UNARY_OP_WITH_SCALAR(+, plus)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_OP_WITH_SCALAR(-, minus)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_OP_WITH_SCALAR(*, mul)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_OP_WITH_SCALAR(/, div)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_OP_WITH_SCALAR(%, mod)


#define WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(func) \
    template <class A, \
        class = std::enable_if_t<is_tensor<std::decay_t<A>>::value>> \
    constexpr auto func(A && a) { \
        return compose_category(a.shape(), \
            compose_ewise_op_result(unary_func_##func(), forward<A>(a))); \
    }

    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(sin)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(cos)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(tan)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(sinh)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(cosh)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(tanh)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(asin)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(acos)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(atan)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(asinh)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(acosh)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(atanh)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(exp)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(exp2)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(log)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(log10)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(log2)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(abs)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(sqrt)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(cbrt)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(ceil)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(floor)
    WHEELS_TS_OVERLOAD_EWISE_UNARY_FUNC(round)


    // ewise_mul
    template <class A, class B,
        class = std::enable_if_t<is_tensor<std::decay_t<A>>::value &&
        is_tensor<std::decay_t<B>>::value >>
    constexpr auto ewise_mul(A && a, B && b) {
        assert(a.shape() == b.shape());
        return compose_category(a.shape(),
            compose_ewise_op_result(binary_op_mul(), forward<A>(a), forward<B>(b)));
    }

    // ewise_div
    template <class A, class B,
        class = std::enable_if_t<is_tensor<std::decay_t<A>>::value &&
        is_tensor<std::decay_t<B>>::value >>
    constexpr auto ewise_div(A && a, B && b) {
        assert(a.shape() == b.shape());
        return compose_category(a.shape(),
            compose_ewise_op_result(binary_op_div(), forward<A>(a), forward<B>(b)));
    }



#define WHEELS_TS_OVERLOAD_EWISE_BINARY_FUNC(func) \
    template <class A, class B, \
        class = std::enable_if_t<is_tensor<std::decay_t<A>>::value &&  \
        is_tensor<std::decay_t<B>>::value>> \
    constexpr auto ewise_##func (A && a, B && b) { \
        assert(a.shape() == b.shape()); \
        return compose_category(a.shape(), \
            compose_ewise_op_result(binary_func_##func(), forward<A>(a), forward<B>(b))); \
    }

    WHEELS_TS_OVERLOAD_EWISE_BINARY_FUNC(max)
    WHEELS_TS_OVERLOAD_EWISE_BINARY_FUNC(min)
    WHEELS_TS_OVERLOAD_EWISE_BINARY_FUNC(pow)
    WHEELS_TS_OVERLOAD_EWISE_BINARY_FUNC(atan2)







    // special shapes
    // vector shape
    template <class CategoryT, bool Writable>
    class vector : public ts_specific_shape_base<CategoryT> {
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
    class vector<CategoryT, true> : public ts_specific_shape_base<CategoryT> {
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
    template <class CategoryT, class ST, class SizeT>
    class ts_specific_shape<CategoryT, tensor_shape<ST, SizeT>>
        : public vector<CategoryT, ts_traits::writable<CategoryT>::value> {};

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






    // matrix shape
    template <class CategoryT, bool Writable>
    class matrix : public ts_specific_shape_base<CategoryT> {
    public:
        constexpr auto rows() const { return size(const_index<0>()); }
        constexpr auto cols() const { return size(const_index<1>()); }
        constexpr auto area() const { return rows() * cols(); }
    };
    template <class CategoryT, class ST, class MT, class NT>
    class ts_specific_shape<CategoryT, tensor_shape<ST, MT, NT>>
        : public matrix<CategoryT, ts_traits::writable<CategoryT>::value> {};

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



    // cube shape
    template <class CategoryT, bool Writable>
    class cube : public ts_specific_shape_base<CategoryT> {
    public:
        constexpr auto volume() const {
            return size(const_index<0>()) * size(const_index<1>()) * size(const_index<2>());
        }
    };
    template <class CategoryT, class ST, class N1T, class N2T, class N3T>
    class ts_specific_shape<CategoryT, tensor_shape<ST, N1T, N2T, N3T>>
        : public cube<CategoryT, ts_traits::writable<CategoryT>::value> {};







    // special value types
    template <class CategoryT>
    class boolean_tensor : public ts_specific_value_type_base<CategoryT> {
    public:
        // conversion to bool value
        constexpr operator bool() const { return all(); }
    };
    template <class CategoryT>
    class ts_specific_value_type<CategoryT, bool>
        : public boolean_tensor<CategoryT> {};







    // dot product
    template <class C1, class C2, bool RInd1, bool RSub1, bool RInd2, bool RSub2>
    constexpr auto dot(const ts_readable<C1, RInd1, RSub1, true> & a,
        const ts_readable<C2, RInd2, RSub2, true> & b) {
        return ewise_mul(a.category(), b.category()).sum();
    }   




    // reshape



    // cat


    // repeat


    // permute


    // 


}