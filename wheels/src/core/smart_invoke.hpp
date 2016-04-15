#pragma once

#include <type_traits>

#include "const_expr.hpp"

#include "smart_invoke_fwd.hpp"

namespace wheels {
namespace details {
// _has_const_expr
template <class T, class... ArgTs>
constexpr auto _has_const_expr(const const_expr_base<T> &, ArgTs &...) {
  return yes();
}
template <class T, class = std::enable_if_t<!is_const_expr<T>::value>,
          class... ArgTs>
constexpr auto _has_const_expr(const T &, ArgTs &... args) {
  return _has_const_expr(args...);
}
constexpr auto _has_const_expr() { return no(); }

// _pass_or_evaluate
template <class T, class TT, class... ArgTs>
constexpr decltype(auto) _pass_or_evaluate(const const_expr_base<T> &,
                                           TT &&expr, ArgTs &&... args) {
  return expr(std::forward<ArgTs>(args)...);
}
template <class T, class = std::enable_if_t<!is_const_expr<T>::value>, class TT,
          class... ArgTs>
constexpr TT &&_pass_or_evaluate(const T &, TT &&expr, ArgTs &&...) {
  return static_cast<TT &&>(expr);
}

// _functor_expr
template <class FunT, class... ArgTs> struct _functor_expr;
template <class FunT, class... ArgTs, class FT, size_t... Is, class... NewArgTs>
constexpr decltype(auto)
_call_functor_expr_seq(const _functor_expr<FunT, ArgTs...> &, FT &&f,
                       const const_ints<size_t, Is...> &,
                       NewArgTs &&... nargs) {
  return std::forward<FT>(f).fun(_pass_or_evaluate(
      std::forward<ArgTs>(std::get<Is>(std::forward<FT>(f).args)),
      std::get<Is>(std::forward<FT>(f).args),
      std::forward<NewArgTs>(nargs)...)...);
}

template <class FunT, class... ArgTs>
struct _functor_expr : const_expr_base<_functor_expr<FunT, ArgTs...>> {
  FunT fun;
  std::tuple<ArgTs...> args;

  constexpr explicit _functor_expr(FunT &&f, ArgTs &&... args)
      : fun(std::forward<FunT>(f)), args(std::forward<ArgTs>(args)...) {}

  template <class... NewArgTs>
  inline decltype(auto) operator()(NewArgTs &&... nargs) & {
    return _call_functor_expr_seq(*this, *this,
                                  make_const_sequence_for<ArgTs...>(),
                                  std::forward<NewArgTs>(nargs)...);
  }
  template <class... NewArgTs>
  constexpr decltype(auto) operator()(NewArgTs &&... nargs) const & {
    return _call_functor_expr_seq(*this, *this,
                                  make_const_sequence_for<ArgTs...>(),
                                  std::forward<NewArgTs>(nargs)...);
  }
  template <class... NewArgTs>
  inline decltype(auto) operator()(NewArgTs &&... nargs) && {
    return _call_functor_expr_seq(*this, std::move(*this),
                                  make_const_sequence_for<ArgTs...>(),
                                  std::forward<NewArgTs>(nargs)...);
  }

  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(fun, args);
  }
};

// _smart_invoke
template <class FunT, class... ArgTs>
constexpr _functor_expr<FunT, ArgTs...>
_smart_invoke(FunT &&fun, yes there_are_const_exprs, ArgTs &&... args) {
  return _functor_expr<FunT, ArgTs...>(std::forward<FunT>(fun),
                                       std::forward<ArgTs>(args)...);
}
template <class FunT, class... ArgTs>
constexpr decltype(auto) _smart_invoke(FunT &&fun, no there_are_const_exprs,
                                       ArgTs &&... args) {
  return fun(std::forward<ArgTs>(args)...);
}
}

// smart_invoke
// migrate const_exprs into arguments
template <class FunT, class... ArgTs>
constexpr decltype(auto) smart_invoke(FunT &&fun, ArgTs &&... args) {
  return details::_smart_invoke(std::forward<FunT>(fun),
                                details::_has_const_expr(args...),
                                std::forward<ArgTs>(args)...);
}
}
