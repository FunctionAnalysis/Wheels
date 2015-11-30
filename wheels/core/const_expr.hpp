#pragma once

#include "constants.hpp"
#include "functors.hpp"

namespace wheels {

    template <class T> struct is_const_expr : no {};

    template <class DerivedT>
    struct const_expr_base {
        constexpr const_expr_base() {}
        constexpr const DerivedT & derived() const { 
            return static_cast<const DerivedT &>(*this); 
        }
    };

    // const_symbol
    template <size_t Idx>
    struct const_symbol : const_expr_base<const_symbol<Idx>> {
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
    struct const_coeff : const_expr_base<const_coeff<T>> {
        static_assert(!is_const_expr<T>::value, "const_coeff should not be nested");
        T val;
        template <class TT>
        constexpr const_coeff(TT && v) : val(forward<TT>(v)) {}
        template <class ... ArgTs>
        constexpr T operator()(const ArgTs & ...) const {
            return val;
        }
        template <class Archive> void serialize(Archive & ar) { ar(val); }
    };
    template <class T> struct is_const_expr<const_coeff<T>> : yes {};
    template <class T> 
    constexpr const_coeff<std::decay_t<T>> as_const_coeff(T && v) {
        return const_coeff<std::decay_t<T>>(forward<T>(v));
    }

    // const_unary_op
    template <class Op, class E>
    struct const_unary_op : const_expr_base<const_unary_op<Op, E>>{
        Op op;
        E e;
        template <class OpT, class T>
        constexpr const_unary_op(OpT && op, T && e) 
            : op(forward<OpT>(op)), e(forward<T>(e)) {}
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
    struct const_binary_op : const_expr_base<const_binary_op<Op, E1, E2>>{
        Op op;
        E1 e1;
        E2 e2;
        template <class OpT, class T1, class T2>
        constexpr const_binary_op(OpT && op, T1 && e1, T2 && e2)
            : op(forward<OpT>(op)), e1(forward<T1>(e1)), e2(forward<T2>(e2)) {}
        template <class ... ArgTs>
        constexpr auto operator()(const ArgTs & ... args) const {
            return op(e1(args...), e2(args...));
        }
        template <class Archive> void serialize(Archive & ar) { ar(op, e1, e2); }
    };
    template <class Op, class E1, class E2>
    struct is_const_expr<const_binary_op<Op, E1, E2>> : yes {};


#define WHEELS_CONST_EXPR_OVERLOAD_UNARY_OP(op, name) \
    template <class E> \
    constexpr auto operator op (const const_expr_base<E> & e) { \
        using _op_t = unary_op_##name; \
        return const_unary_op<_op_t, E> (_op_t(), e.derived()); \
    }
    WHEELS_CONST_EXPR_OVERLOAD_UNARY_OP(-, minus)


#define WHEELS_CONST_EXPR_OVERLOAD_BINARY_OP(op, name) \
    template <class E1, class E2>  \
    constexpr auto operator op (const const_expr_base<E1> & e1, const const_expr_base<E2> & e2) { \
        using _op_t = binary_op_##name; \
        return const_binary_op<_op_t, E1, E2> (_op_t(), e1.derived(), e2.derived()); \
    } \
    template <class E1, class E2, class = \
        std::enable_if_t<!(is_const_expr<E2>::value)>, \
        class = void> \
    constexpr auto operator op (const const_expr_base<E1> & e1, const E2 & e2) { \
        return e1.derived() op as_const_coeff(e2); \
    } \
    template <class E1, class E2, class = \
        std::enable_if_t<!(is_const_expr<E1>::value)>, \
        class = void, class = void> \
    constexpr auto operator op (const E1 & e1, const const_expr_base<E2> & e2) { \
        return as_const_coeff(e1) op e2.derived(); \
    } 

    WHEELS_CONST_EXPR_OVERLOAD_BINARY_OP(+, plus)
    WHEELS_CONST_EXPR_OVERLOAD_BINARY_OP(-, minus)
    WHEELS_CONST_EXPR_OVERLOAD_BINARY_OP(*, mul)
    WHEELS_CONST_EXPR_OVERLOAD_BINARY_OP(/, div)
    WHEELS_CONST_EXPR_OVERLOAD_BINARY_OP(%, mod)


}