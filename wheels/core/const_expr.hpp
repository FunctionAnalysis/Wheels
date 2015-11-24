#pragma once

#include "constants.hpp"
#include "operators.hpp"

namespace wheels {

    template <class T> struct is_const_expr : no {};

    // const_symbol
    template <size_t Idx>
    struct const_symbol {
        constexpr const_symbol() {}
        template <class ... ArgTs>
        constexpr auto operator()(const ArgTs & ... args) const {
            return std::get<Idx>(std::forward_as_tuple(args...));
        }
        template <class Archive> void serialize(Archive &) {}
    };
    template <size_t Idx> struct is_const_expr<const_symbol<Idx>> : yes {};

    namespace literals {
        // ""_symbol
        template <char ... Cs>
        constexpr auto operator "" _symbol() {
            return const_symbol<details::_parse_int<size_t, Cs...>::value>();
        }
    }


    // const_coeff
    template <class T>
    struct const_coeff {
        static_assert(!is_const_expr<T>::value, "const_coeff should not be nested");
        T val;
        constexpr const_coeff(const T & v) : val(v) {}
        template <class ... ArgTs>
        constexpr T operator()(const ArgTs & ...) const {
            return val;
        }
        template <class Archive> void serialize(Archive & ar) { ar(val); }
    };
    template <class T> struct is_const_expr<const_coeff<T>> : yes {};
    template <class T> 
    constexpr const_coeff<T> as_const_coeff(const T & v) {
        return const_coeff<T>(v);
    }

    // const_unary_op
    template <class Op, class E>
    struct const_unary_op {
        Op op;
        E e;
        constexpr const_unary_op(const Op & op, const E & e) : op(op), e(e) {}
        template <class ... ArgTs>
        constexpr auto operator()(const ArgTs & ... args) const {
            return op(e(args...));
        }
        template <class Archive> void serialize(Archive & ar) { ar(op, e); }
    };
    template <class Op, class E> 
    struct is_const_expr<const_unary_op<Op, E>> : yes {};

    // const_binary_op
    template <class Op, class E1, class E2>
    struct const_binary_op {
        Op op;
        E1 e1;
        E2 e2;
        constexpr const_binary_op(const Op & op, const E1 & e1, const E2 & e2)
            : op(op), e1(e1), e2(e2) {}
        template <class ... ArgTs>
        constexpr auto operator()(const ArgTs & ... args) const {
            return op(e1(args...), e2(args...));
        }
        template <class Archive> void serialize(Archive & ar) { ar(op, e1, e2); }
    };
    template <class Op, class E1, class E2>
    struct is_const_expr<const_binary_op<Op, E1, E2>> : yes {};


#define WHEELS_CONST_EXPR_OVERLOAD_UNARY_OP(op, name) \
    template <class E, class = std::enable_if_t<is_const_expr<E>::value>> \
    constexpr auto operator op (const E & e) { \
        using _op_t = unary_op_##name; \
        return const_unary_op<_op_t, E> (_op_t(), e); \
    }
    WHEELS_CONST_EXPR_OVERLOAD_UNARY_OP(-, minus)


#define WHEELS_CONST_EXPR_OVERLOAD_BINARY_OP(op, name) \
    template <class E1, class E2, class =  \
        std::enable_if_t<is_const_expr<E1>::value && is_const_expr<E2>::value>>  \
    constexpr auto operator op (const E1 & e1, const E2 & e2) { \
        using _op_t = binary_op_##name; \
        return const_binary_op<_op_t, E1, E2> (_op_t(), e1, e2); \
    } \
    template <class E1, class E2, class = \
        std::enable_if_t<is_const_expr<E1>::value && !(is_const_expr<E2>::value)>, \
        class = void> \
    constexpr auto operator op (const E1 & e1, const E2 & e2) { \
        return e1 op as_const_coeff(e2); \
    } \
    template <class E1, class E2, class = \
        std::enable_if_t<!(is_const_expr<E1>::value) && is_const_expr<E2>::value>, \
        class = void, class = void> \
    constexpr auto operator op (const E1 & e1, const E2 & e2) { \
        return as_const_coeff(e1) op e2; \
    } 

    WHEELS_CONST_EXPR_OVERLOAD_BINARY_OP(+, plus)
    WHEELS_CONST_EXPR_OVERLOAD_BINARY_OP(-, minus)
    WHEELS_CONST_EXPR_OVERLOAD_BINARY_OP(*, mul)
    WHEELS_CONST_EXPR_OVERLOAD_BINARY_OP(/, div)
    WHEELS_CONST_EXPR_OVERLOAD_BINARY_OP(%, mod)


}