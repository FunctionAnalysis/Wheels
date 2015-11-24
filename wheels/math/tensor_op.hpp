#pragma once

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
        constexpr const T & element_at_subs(const eye<T> & a, const SubTs & ... subs) {
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
            return a.op(std::get<Is>(a.inputs).at_index(index) ...);
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
            return a.op(std::get<Is>(a.inputs).at_subs(subs ...) ...);
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
    template <class A, class = std::enable_if_t<details::_all_tensor_layouts<A>::value>>  \
    constexpr auto operator op (A && a) { \
        return compose_tensor_layout(a.shape(), tdp::compose_ewise_op(unary_op_##name(), forward<A>(a))); \
    }

    WHEELS_TENSOR_OVERLOAD_UNARY_OP(+, plus)
    WHEELS_TENSOR_OVERLOAD_UNARY_OP(-, minus)
    WHEELS_TENSOR_OVERLOAD_UNARY_OP(!, not)


#define WHEELS_TENSOR_OVERLOAD_BINARY_OP(op, name) \
    template <class A, class B, class = std::enable_if_t<details::_all_tensor_layouts<A, B>::value>> \
    constexpr auto operator op (A && a, B && b) { \
        assert(a.shape() == b.shape()); \
        return compose_tensor_layout(a.shape(),  \
            tdp::compose_ewise_op(binary_op_##name(), forward<A>(a), forward<B>(b))); \
    }

    WHEELS_TENSOR_OVERLOAD_BINARY_OP(+, plus)
    WHEELS_TENSOR_OVERLOAD_BINARY_OP(-, minus)


#define WHEELS_TENSOR_OVERLOAD_UNARY_OP_WITH(op, name) \
    template <class A, class B, class = \
        std::enable_if_t<details::_all_tensor_layouts<A>::value &&  \
        !details::_all_tensor_layouts<B>::value>, class = void> \
    constexpr auto operator op (A && a, B && b) { \
        return compose_tensor_layout(a.shape(), \
            tdp::compose_ewise_op(unary_op_##name##_with<std::decay_t<B>>(forward<B>(b)), forward<A>(a))); \
    } \
    template <class A, class B, class = \
        std::enable_if_t<details::_all_tensor_layouts<B>::value && \
        !details::_all_tensor_layouts<A>::value>, class = void, class = void> \
        constexpr auto operator op (A && a, B && b) { \
        return compose_tensor_layout(b.shape(), \
            tdp::compose_ewise_op(unary_op_##name##_with_<std::decay_t<A>>(forward<A>(a)), forward<B>(b))); \
    }

    WHEELS_TENSOR_OVERLOAD_UNARY_OP_WITH(+, plus)
    WHEELS_TENSOR_OVERLOAD_UNARY_OP_WITH(-, minus)
    WHEELS_TENSOR_OVERLOAD_UNARY_OP_WITH(*, mul)
    WHEELS_TENSOR_OVERLOAD_UNARY_OP_WITH(/, div)
    WHEELS_TENSOR_OVERLOAD_UNARY_OP_WITH(%, mod)


}
