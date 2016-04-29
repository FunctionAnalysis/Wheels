#pragma once

#include <type_traits>

#include "smart_invoke_fwd.hpp"

#include "const_expr.hpp"

namespace wheels {

template <class T> struct tensor_core;
namespace details {

// _pass_or_evaluate
template <class T, class TT, class... ArgTs>
constexpr auto _pass_or_evaluate_imp(const const_expr_base<T> &, TT &&expr,
                                     ArgTs &&... args)
    -> decltype(expr(std::forward<ArgTs>(args)...)) {
  return expr(std::forward<ArgTs>(args)...);
}
template <class T, class TT, class... ArgTs>
constexpr TT &&_pass_or_evaluate_imp(const tensor_core<T> &, TT &&expr,
                                     ArgTs &&...) {
  return static_cast<TT &&>(expr);
}
template <class T, class TT, class... ArgTs>
constexpr TT &&_pass_or_evaluate_imp(const category::other<T> &, TT &&expr,
                                     ArgTs &&...) {
  return static_cast<TT &&>(expr);
}
template <class T, class... ArgTs>
constexpr decltype(auto) _pass_or_evaluate(T &&expr, ArgTs &&... args) {
  return _pass_or_evaluate_imp(category::identify(expr), std::forward<T>(expr),
                               std::forward<ArgTs>(args)...);
}

//// _functor_expr
// template <class FunT, class... ArgTs> struct _functor_expr;
// template <class FunT, class... ArgTs, class FT, size_t... Is, class...
// NewArgTs>
// constexpr decltype(auto)
//_call_functor_expr_seq(const _functor_expr<FunT, ArgTs...> &, FT &&f,
//                       const const_ints<size_t, Is...> &,
//                       NewArgTs &&... nargs) {
//  return std::forward<FT>(f).fun(_pass_or_evaluate(
//      std::forward<ArgTs>(std::get<Is>(std::forward<FT>(f).args)),
//      std::forward<NewArgTs>(nargs)...)...);
//}

//template <class FunT, class... ArgTs>
//class _functor_expr : const_expr_base<_functor_expr<FunT, ArgTs...>> {
//public:
//  constexpr explicit _functor_expr(FunT f, ArgTs &&... args)
//      : fun(f), args(std::forward<ArgTs>(args)...) {}
//
//  template <class... NewArgTs>
//  inline auto operator()(NewArgTs &&... nargs) const {
//    return const_cast<_functor_expr &>(*this)._call_seq(
//        make_const_sequence_for<ArgTs...>(), std::forward<NewArgTs>(nargs)...);
//  }
//
//private:
//  template <size_t... Is, class... NewArgTs>
//  inline auto _call_seq(const const_ints<size_t, Is...> &,
//                        NewArgTs &&... nargs) {
//    return this->fun(
//        _pass_or_evaluate(std::forward<ArgTs>(std::get<Is>(this->args)),
//                          std::forward<NewArgTs>(nargs)...)...);
//  }
//
//public:
//  FunT fun;
//  std::tuple<ArgTs...> args;
//};

// _smart_invoke
template <class FunT, class... ArgTs>
constexpr auto _smart_invoke(FunT fun, yes there_are_const_exprs,
                             ArgTs &&... args) {
  //return _functor_expr<FunT, ArgTs...>(fun, std::forward<ArgTs>(args)...);
  return make_const_call_list(fun,
                              as_const_coeff(std::forward<ArgTs>(args))...);
}
template <class FunT, class... ArgTs>
constexpr decltype(auto) _smart_invoke(FunT fun, no there_are_const_exprs,
                                       ArgTs &&... args) {
  return fun(std::forward<ArgTs>(args)...);
}
}

// smart_invoke
// migrate const_exprs into arguments
template <class FunT, class... ArgTs>
constexpr decltype(auto) smart_invoke(FunT fun, ArgTs &&... args) {
  return details::_smart_invoke(fun, has_const_expr(args...),
                                std::forward<ArgTs>(args)...);
}
}
