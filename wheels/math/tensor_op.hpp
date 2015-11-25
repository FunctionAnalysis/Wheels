#pragma once

#include "../core/macros.hpp"

#include "tensor_data.hpp"
#include "tensor.hpp"

namespace wheels {

    // constant
    namespace tdp {
        template <class T>
        struct constant {
            using value_type = T;
            T val;
            template <class TT>
            constexpr constant(TT && v) : val(forward<TT>(v)) {}
            template <class Archiver>
            void serialize(Archiver & ar) {
                ar(_val);
            }
        };

        // accessing elements
        template <class T>
        struct is_element_readable_at_index<constant<T>> : yes {};

        template <class T, class IndexT>
        constexpr const T & element_at_index(const constant<T> & a, const IndexT & index) {
            return a.val;
        }

        template <class T>
        struct is_element_readable_at_subs<constant<T>> : yes {};

        template <class T, class ... SubTs>
        constexpr const T & element_at_subs(const constant<T> & a, const SubTs & ...) {
            return a.val;
        }
    }

    template <class T, class ST, class ... SizeTs>
    constexpr auto constants(const tensor_shape<ST, SizeTs...> & shape, T && val) {
        return compose_tensor(shape, tdp::constant<std::decay_t<T>>(forward<T>(val)));
    }

    template <class T = double, class ST, class ... SizeTs>
    constexpr auto zeros(const tensor_shape<ST, SizeTs...> & shape) {
        return compose_tensor(shape, tdp::constant<T>(0));
    }

    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto zeros(const SizeTs & ... sizes) {
        return compose_tensor(make_shape<ST>(sizes ...), tdp::constant<T>(0));
    }

    template <class T = double, class ST, class ... SizeTs>
    constexpr auto ones(const tensor_shape<ST, SizeTs...> & shape) {
        return compose_tensor(shape, tdp::constant<T>(1));
    }

    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto ones(const SizeTs & ... sizes) {
        return compose_tensor(make_shape<ST>(sizes ...), tdp::constant<T>(1));
    }



    // eye
    namespace tdp {
        template <class T>
        struct eye {
            using value_type = T;
            constexpr eye() {}
            template <class Archiver>
            void serialize(Archiver & ar) {}
        };

        // accessing elements
        template <class T>
        struct is_element_readable_at_subs<eye<T>> : yes {};

        template <class T, class ... SubTs>
        constexpr auto element_at_subs(const eye<T> & a, const SubTs & ... subs) {
            static_assert(sizeof...(SubTs) > 1, "shape rank must be over 1");
            return all_same(subs ...) ? 1 : 0;
        }
    }

    template <class T = double, class ST, class ... SizeTs>
    constexpr auto eye(const tensor_shape<ST, SizeTs ...> & shape) {
        return compose_tensor(shape, tdp::eye<T>());
    }

    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto eye(const SizeTs & ... sizes) {
        return compose_tensor(make_shape<ST>(sizes ...), tdp::eye<T>());
    }


    // iota
    namespace tdp {
        template <class T>
        struct iota {
            using value_type = T;
            constexpr iota() {}
            template <class Archiver>
            void serialize(Archiver & ar) {}
        };
        // accessing elements
        template <class T> struct is_element_readable_at_index<iota<T>> : yes {};
        template <class T, class IndexT>
        constexpr T element_at_index(const iota<T> & a, const IndexT & index) {
            return T(index);
        }
    }

    template <class T = double, class ST, class ... SizeTs>
    constexpr auto iota(const tensor_shape<ST, SizeTs ...> & shape) {
        return compose_tensor(shape, tdp::iota<T>());
    }

    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto iota(const SizeTs & ... sizes) {
        return compose_tensor(make_shape<ST>(sizes ...), tdp::iota<T>());
    }



    // ewise op
    namespace tdp {
        template <class T, class OpT, class ... InputTs>
        struct ewise_op {
            using value_type = T;
            OpT op;
            std::tuple<InputTs ...> inputs;
            template <class OpTT, class ... InputTTs>
            constexpr explicit ewise_op(OpTT && o, InputTTs && ... ins) 
                : op(forward<OpTT>(o)), inputs(forward<InputTTs>(ins) ...) {}
            template <class Archiver>
            void serialize(Archiver & ar) {
                ar(op, inputs);
            }
        };

        // accessing elements
        template <class T, class OpT, class ... InputTs>
        struct is_element_readable_at_index<ewise_op<T, OpT, InputTs ...>> : yes {};

        template <class T, class OpT, class ... InputTs, class IndexT, size_t ... Is>
        constexpr decltype(auto) _element_at_index_seq(const ewise_op<T, OpT, InputTs ...> & a,
            const IndexT & index, const_ints<size_t, Is ...>) {
            return a.op(std::get<Is>(a.inputs).at_index_const(index) ...);
        }
        template <class T, class OpT, class ... InputTs, class IndexT>
        constexpr decltype(auto) element_at_index(const ewise_op<T, OpT, InputTs ...> & a, const IndexT & index) {
            return _element_at_index_seq(a, index, make_const_sequence(const_size<sizeof...(InputTs)>()));
        }

        template <class T, class OpT, class ... InputTs>
        struct is_element_readable_at_subs<ewise_op<T, OpT, InputTs ...>> : yes {};

        template <class T, class OpT, class ... InputTs, class ... SubTs, size_t ... Is>
        constexpr decltype(auto) _element_at_subs_seq(const ewise_op<T, OpT, InputTs ...> & a,
            const_ints<size_t, Is ...>, const SubTs & ... subs) {
            return a.op(std::get<Is>(a.inputs).at_subs_const(subs ...) ...);
        }
        template <class T, class OpT, class ... InputTs, class ... SubTs>
        constexpr decltype(auto) element_at_subs(const ewise_op<T, OpT, InputTs ...> & a, const SubTs & ... subs) {
            return _element_at_subs_seq(a, make_const_sequence(const_size<sizeof...(InputTs)>()), subs ...);
        }

        template <class OpT, class ... InputTs>
        constexpr auto compose_ewise_op(OpT && op, InputTs && ... inputs) {
            using result_t = std::decay_t<decltype(op(inputs.at_index_const(0) ...))>;
            return ewise_op<result_t, OpT, InputTs...>(forward<OpT>(op), forward<InputTs>(inputs) ...);
        }
    }



#define WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_OP(op, name) \
    template <class A, \
        class = std::enable_if_t<is_tensor<std::decay_t<A>>::value>, \
        wheels_distinguish_1>  \
    constexpr auto operator op (A && a) { \
        return compose_tensor(a.shape(), tdp::compose_ewise_op(unary_op_##name(), forward<A>(a))); \
    }

    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_OP(+, plus)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_OP(-, minus)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_OP(!, not)


#define WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_OP(op, name) \
    template <class A, class B, \
        class = std::enable_if_t<is_tensor<std::decay_t<A>>::value && is_tensor<std::decay_t<B>>::value>, \
        wheels_distinguish_3> \
    constexpr auto operator op (A && a, B && b) { \
        assert(a.shape() == b.shape()); \
        return compose_tensor(a.shape(),  \
            tdp::compose_ewise_op(binary_op_##name(), forward<A>(a), forward<B>(b))); \
    }

    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_OP(+, plus)
    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_OP(-, minus)
    
    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_OP(==, eq)
    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_OP(!=, neq)
    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_OP(<, lt)
    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_OP(<=, lte)
    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_OP(>, gt)
    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_OP(>=, gte)

    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_OP(&&, and)
    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_OP(||, or)
    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_OP(&, bitwise_and)
    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_OP(|, bitwise_or)


#define WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_OP_WITH_SCALAR(op, name) \
    template <class A, class B, class = \
        std::enable_if_t<\
         is_tensor<std::decay_t<A>>::value &&  \
        !is_tensor<std::decay_t<B>>::value && \
        !is_const_expr<std::decay_t<B>>::value>, wheels_distinguish_4> \
    constexpr auto operator op (A && a, B && b) { \
        return compose_tensor(a.shape(), \
            tdp::compose_ewise_op(unary_op_##name##_with<std::decay_t<B>>(forward<B>(b)), forward<A>(a))); \
    } \
    template <class A, class B, class = \
        std::enable_if_t<\
         is_tensor<std::decay_t<B>>::value && \
        !is_tensor<std::decay_t<A>>::value && \
        !is_const_expr<std::decay_t<A>>::value>, wheels_distinguish_5> \
        constexpr auto operator op (A && a, B && b) { \
        return compose_tensor(b.shape(), \
            tdp::compose_ewise_op(unary_op_##name##_with_<std::decay_t<A>>(forward<A>(a)), forward<B>(b))); \
    }

    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_OP_WITH_SCALAR(+, plus)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_OP_WITH_SCALAR(-, minus)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_OP_WITH_SCALAR(*, mul)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_OP_WITH_SCALAR(/, div)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_OP_WITH_SCALAR(%, mod)


#define WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(func) \
    template <class A, \
        class = std::enable_if_t<is_tensor<std::decay_t<A>>::value>> \
    constexpr auto func(A && a) { \
        return compose_tensor(a.shape(), \
            tdp::compose_ewise_op(unary_func_##func(), forward<A>(a))); \
    }

    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(sin)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(cos)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(tan)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(sinh)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(cosh)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(tanh)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(asin)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(acos)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(atan)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(asinh)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(acosh)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(atanh)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(exp)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(exp2)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(log)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(log10)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(log2)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(abs)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(sqrt)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(cbrt)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(ceil)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(floor)
    WHEELS_TENSOR_OVERLOAD_EWISE_UNARY_FUNC(round)


    // ewise_mul
    template <class A, class B, 
        class = std::enable_if_t<is_tensor<std::decay_t<A>>::value && 
        is_tensor<std::decay_t<B>>::value>>
    constexpr auto ewise_mul (A && a, B && b) {
        assert(a.shape() == b.shape());
        return compose_tensor(a.shape(),
            tdp::compose_ewise_op(binary_op_mul(), forward<A>(a), forward<B>(b)));
    }

    // ewise_div
    template <class A, class B, 
        class = std::enable_if_t<is_tensor<std::decay_t<A>>::value && 
        is_tensor<std::decay_t<B>>::value>>
    constexpr auto ewise_div (A && a, B && b) {
        assert(a.shape() == b.shape());
        return compose_tensor(a.shape(),
            tdp::compose_ewise_op(binary_op_div(), forward<A>(a), forward<B>(b)));
    }



#define WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_FUNC(func) \
    template <class A, class B, \
        class = std::enable_if_t<is_tensor<std::decay_t<A>>::value &&  \
        is_tensor<std::decay_t<B>>::value>> \
    constexpr auto ewise_##func (A && a, B && b) { \
        assert(a.shape() == b.shape()); \
        return compose_tensor(a.shape(), \
            tdp::compose_ewise_op(binary_func_##func(), forward<A>(a), forward<B>(b))); \
    }

    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_FUNC(max)
    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_FUNC(min)
    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_FUNC(pow)
    WHEELS_TENSOR_OVERLOAD_EWISE_BINARY_FUNC(atan2)


}
