#pragma once

#include "object_fwd.hpp"

namespace wheels {

// const_expr_base
template <class T> struct const_expr_base;

// const_symbol
template <size_t Idx> struct const_symbol;

namespace literals {
// ""_symbol
template <char... Cs> constexpr auto operator"" _symbol();
}

template <class T> struct const_coeff;
template <class T> constexpr const_coeff<std::decay_t<T>> as_const_coeff(T &&v);

template <class Op, class E> struct const_unary_op;
template <class Op, class E>
constexpr const_unary_op<Op, E> make_unary_op_expr(const Op &op, E &&e);

template <class Op, class E1, class E2> struct const_binary_op;

template <class Op, class E1, class E2>
constexpr const_binary_op<Op, E1, E2> make_binary_op_expr(const Op &op, E1 &&e1,
                                                          E2 &&e2);

template <class OpT> struct func_base;
template <class OpT, class T>
constexpr auto overload_as(const func_base<OpT> &, const const_expr_base<T> &);

template <class OpT, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &, const const_expr_base<T1> &,
                           const const_expr_base<T2> &);

template <class OpT, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &, const const_expr_base<T1> &,
                           const category::other<T2> &);

template <class OpT, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &, const category::other<T1> &,
                           const const_expr_base<T2> &);

template <class FunT, class... RecordedArgTs> struct const_call_list;
}
