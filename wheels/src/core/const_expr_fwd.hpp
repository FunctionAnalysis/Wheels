#pragma once

#include "object_fwd.hpp"

namespace wheels {

// const_expr_base
template <class T> struct const_expr_base;

// invoke_const_expr
template <class E, class... ArgTs>
decltype(auto) invoke_const_expr(E &&e, ArgTs &&... args);

// const_symbol
template <size_t Idx> struct const_symbol;

namespace literals {
// ""_symbol
template <char... Cs> constexpr auto operator"" _symbol();
}

template <class T> struct const_coeff;
template <class T> constexpr decltype(auto) as_const_coeff(T &&v);


template <class FunT, class... RecordedArgTs> struct const_call_list;
template <class FunT, class... RecordedExprArgTs>
constexpr auto make_const_call_list(FunT&& f, RecordedExprArgTs &&... as);

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

// has_const_expr
template <class... ArgTs> constexpr auto has_const_expr(const ArgTs &... args);

// smart_invoke
template <class FunT, class... ArgTs>
constexpr decltype(auto) smart_invoke(FunT&& fun, ArgTs &&... args);
}
