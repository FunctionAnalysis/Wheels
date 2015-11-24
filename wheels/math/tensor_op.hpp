#pragma once

#include "../core/macros.hpp"

#include "tensor_data.hpp"
#include "tensor.hpp"

namespace wheels {

    // constant
    namespace tdp {
        template <class T>
        struct constant {
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
        return compose_tensor_layout(shape, tdp::constant<std::decay_t<T>>(forward<T>(val)));
    }

    template <class T = double, class ST, class ... SizeTs>
    constexpr auto zeros(const tensor_shape<ST, SizeTs...> & shape) {
        return compose_tensor_layout(shape, tdp::constant<T>(0));
    }

    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto zeros(const SizeTs & ... sizes) {
        return compose_tensor_layout(make_shape<ST>(sizes ...), tdp::constant<T>(0));
    }

    template <class T = double, class ST, class ... SizeTs>
    constexpr auto ones(const tensor_shape<ST, SizeTs...> & shape) {
        return compose_tensor_layout(shape, tdp::constant<T>(1));
    }

    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto ones(const SizeTs & ... sizes) {
        return compose_tensor_layout(make_shape<ST>(sizes ...), tdp::constant<T>(1));
    }



    // eye
    namespace tdp {
        template <class T>
        struct eye {
            constexpr eye() {}
            template <class Archiver>
            void serialize(Archiver & ar) {}
        };

        // accessing elements
        template <class T>
        struct is_element_readable_at_subs<eye<T>> : yes {};

        template <class T, class ... SubTs>
        constexpr auto element_at_subs(const eye<T> & a, const SubTs & ... subs) {
            static_assert(sizeof...(SubTs) > 1, "shape degree must be over 1");
            return all_same(subs ...) ? 1 : 0;
        }
    }

    template <class T = double, class ST, class ... SizeTs>
    constexpr auto eye(const tensor_shape<ST, SizeTs ...> & shape) {
        return compose_tensor_layout(shape, tdp::eye<T>());
    }

    template <class T = double, class ST = size_t, class ... SizeTs>
    constexpr auto eye(const SizeTs & ... sizes) {
        return compose_tensor_layout(make_shape<ST>(sizes ...), tdp::eye<T>());
    }



    // iota
    namespace tdp {
        struct iota {
            constexpr iota() {}
            template <class Archiver>
            void serialize(Archiver & ar) {}
        };
        // accessing elements
        template <> struct is_element_readable_at_index<iota> : yes {};
        template <class IndexT>
        constexpr IndexT && element_at_index(const iota & a, IndexT && index) {
            return static_cast<IndexT>(index);
        }
    }

    template <class ST, class ... SizeTs>
    constexpr auto iota(const tensor_shape<ST, SizeTs ...> & shape) {
        return compose_tensor_layout(shape, tdp::iota());
    }

    template <class ST = size_t, class ... SizeTs>
    constexpr auto iota(const SizeTs & ... sizes) {
        return compose_tensor_layout(make_shape<ST>(sizes ...), tdp::iota());
    }



    // ewise op
    namespace tdp {
        template <class OpT, class ... InputTs>
        struct ewise_op {
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
        template <class OpT, class ... InputTs>
        struct is_element_readable_at_index<ewise_op<OpT, InputTs ...>> : yes {};

        template <class OpT, class ... InputTs, class IndexT, size_t ... Is>
        constexpr decltype(auto) _element_at_index_seq(const ewise_op<OpT, InputTs ...> & a,
            const IndexT & index, const_ints<size_t, Is ...>) {
            return a.op(std::get<Is>(a.inputs).method_read_element().at_index(index) ...);
        }
        template <class OpT, class ... InputTs, class IndexT>
        constexpr decltype(auto) element_at_index(const ewise_op<OpT, InputTs ...> & a, const IndexT & index) {
            return _element_at_index_seq(a, index, make_const_sequence(const_size<sizeof...(InputTs)>()));
        }

        template <class OpT, class ... InputTs>
        struct is_element_readable_at_subs<ewise_op<OpT, InputTs ...>> : yes {};

        template <class OpT, class ... InputTs, class ... SubTs, size_t ... Is>
        constexpr decltype(auto) _element_at_subs_seq(const ewise_op<OpT, InputTs ...> & a,
            const_ints<size_t, Is ...>, const SubTs & ... subs) {
            return a.op(std::get<Is>(a.inputs).method_read_element().at_subs(subs ...) ...);
        }
        template <class OpT, class ... InputTs, class ... SubTs>
        constexpr decltype(auto) element_at_subs(const ewise_op<OpT, InputTs ...> & a, const SubTs & ... subs) {
            return _element_at_subs_seq(a, make_const_sequence(const_size<sizeof...(InputTs)>()), subs ...);
        }

        template <class OpT, class ... InputTs>
        constexpr ewise_op<OpT, InputTs...> compose_ewise_op(OpT && op, InputTs && ... inputs) {
            return ewise_op<OpT, InputTs...>(forward<OpT>(op), forward<InputTs>(inputs) ...);
        }
    }

    namespace details {
        template <class ... Ts>
        struct _all_tensor_layouts {
            static constexpr bool value =
                const_ints<bool, is_tensor_layout<std::decay_t<Ts>>::value ...>::all();
        };
    }


#define WHEELS_TENSOR_OVERLOAD_UNARY_OP(op, name) \
    template <class A, class = std::enable_if_t<is_tensor_layout<std::decay_t<A>>::value>, class = void>  \
    constexpr auto operator op (A && a) { \
        return compose_tensor_layout(a.shape(), tdp::compose_ewise_op(unary_op_##name(), forward<A>(a))); \
    }

    WHEELS_TENSOR_OVERLOAD_UNARY_OP(+, plus)
    WHEELS_TENSOR_OVERLOAD_UNARY_OP(-, minus)
    WHEELS_TENSOR_OVERLOAD_UNARY_OP(!, not)


#define WHEELS_TENSOR_OVERLOAD_BINARY_OP(op, name) \
    template <class A, class B, \
        class = std::enable_if_t<is_tensor_layout<std::decay_t<A>>::value && is_tensor_layout<std::decay_t<B>>::value>, \
        wheels_distinguish_3> \
    constexpr auto operator op (A && a, B && b) { \
        assert(a.shape() == b.shape()); \
        return compose_tensor_layout(a.shape(),  \
            tdp::compose_ewise_op(binary_op_##name(), forward<A>(a), forward<B>(b))); \
    }

    WHEELS_TENSOR_OVERLOAD_BINARY_OP(+, plus)
    WHEELS_TENSOR_OVERLOAD_BINARY_OP(-, minus)
    
    WHEELS_TENSOR_OVERLOAD_BINARY_OP(==, eq)
    WHEELS_TENSOR_OVERLOAD_BINARY_OP(!=, neq)
    WHEELS_TENSOR_OVERLOAD_BINARY_OP(<, lt)
    WHEELS_TENSOR_OVERLOAD_BINARY_OP(<=, lte)
    WHEELS_TENSOR_OVERLOAD_BINARY_OP(>, gt)
    WHEELS_TENSOR_OVERLOAD_BINARY_OP(>=, gte)

    WHEELS_TENSOR_OVERLOAD_BINARY_OP(&&, and)
    WHEELS_TENSOR_OVERLOAD_BINARY_OP(||, or)
    WHEELS_TENSOR_OVERLOAD_BINARY_OP(&, bitwise_and)
    WHEELS_TENSOR_OVERLOAD_BINARY_OP(|, bitwise_or)


#define WHEELS_TENSOR_OVERLOAD_UNARY_OP_WITH_SCALAR(op, name) \
    template <class A, class B, class = \
        std::enable_if_t<\
         is_tensor_layout<std::decay_t<A>>::value &&  \
        !is_tensor_layout<std::decay_t<B>>::value && \
        !is_const_expr<std::decay_t<B>>::value>, wheels_distinguish_4> \
    constexpr auto operator op (A && a, B && b) { \
        return compose_tensor_layout(a.shape(), \
            tdp::compose_ewise_op(unary_op_##name##_with<std::decay_t<B>>(forward<B>(b)), forward<A>(a))); \
    } \
    template <class A, class B, class = \
        std::enable_if_t<\
         is_tensor_layout<std::decay_t<B>>::value && \
        !is_tensor_layout<std::decay_t<A>>::value && \
        !is_const_expr<std::decay_t<A>>::value>, wheels_distinguish_5> \
        constexpr auto operator op (A && a, B && b) { \
        return compose_tensor_layout(b.shape(), \
            tdp::compose_ewise_op(unary_op_##name##_with_<std::decay_t<A>>(forward<A>(a)), forward<B>(b))); \
    }

    WHEELS_TENSOR_OVERLOAD_UNARY_OP_WITH_SCALAR(+, plus)
    WHEELS_TENSOR_OVERLOAD_UNARY_OP_WITH_SCALAR(-, minus)
    WHEELS_TENSOR_OVERLOAD_UNARY_OP_WITH_SCALAR(*, mul)
    WHEELS_TENSOR_OVERLOAD_UNARY_OP_WITH_SCALAR(/, div)
    WHEELS_TENSOR_OVERLOAD_UNARY_OP_WITH_SCALAR(%, mod)


#define WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(func) \
    template <class A, class = std::enable_if_t<is_tensor_layout<std::decay_t<A>>::value>> \
    constexpr auto func(A && a) { \
        return compose_tensor_layout(a.shape(), \
            tdp::compose_ewise_op(unary_func_##func(), forward<A>(a))); \
    }

    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(sin)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(cos)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(tan)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(sinh)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(cosh)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(tanh)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(asin)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(acos)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(atan)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(asinh)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(acosh)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(atanh)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(exp)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(log)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(log10)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(log2)
    WHEELS_TENSOR_OVERLOAD_UNARY_FUNC(abs)


}
