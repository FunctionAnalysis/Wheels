#pragma once

#include "overloads.hpp"

namespace wheels {

struct category_const_expr {};

// const_expr_base
template <class T> struct const_expr_base {
  constexpr const T &derived() const { return static_cast<const T &>(*this); }
  T &derived() { return static_cast<T &>(*this); }
};

template <class T, class OpT>
constexpr auto category_for_overloading(const const_expr_base<T> &,
                                        const common_func<OpT> &) {
  return category_const_expr();
}

// is_const_expr
template <class T>
struct is_const_expr : std::is_base_of<const_expr_base<T>, T> {};

// const_symbol
template <size_t Idx> struct const_symbol : const_expr_base<const_symbol<Idx>> {
  constexpr const_symbol() {}
  template <class... ArgTs> constexpr auto operator()(ArgTs &&... args) const {
    return std::get<Idx>(std::forward_as_tuple(forward<ArgTs>(args)...));
  }
};

namespace literals {
// ""_symbol
template <char... Cs> constexpr auto operator"" _symbol() {
  return const_symbol<details::_parse_int<size_t, Cs...>::value>();
}
}

// const_coeff
template <class T> struct const_coeff : const_expr_base<const_coeff<T>> {
  T val;
  template <class TT> constexpr const_coeff(TT &&v) : val(forward<TT>(v)) {}
  template <class... ArgTs> constexpr T operator()(ArgTs &&...) const {
    return val;
  }
  template <class V> decltype(auto) fields(V &&visitor) { return visitor(val); }
};

template <class T>
constexpr const_coeff<std::decay_t<T>> as_const_coeff(T &&v) {
  return const_coeff<std::decay_t<T>>(forward<T>(v));
}

// const_unary_op
template <class Op, class E>
struct const_unary_op : const_expr_base<const_unary_op<Op, E>> {
  Op op;
  E e;
  template <class OpT, class T>
  constexpr const_unary_op(OpT &&op, T &&e)
      : op(forward<OpT>(op)), e(forward<T>(e)) {}
  template <class... ArgTs> constexpr auto operator()(ArgTs &&... args) const {
    return op(e(forward<ArgTs>(args)...));
  }
  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(op, e);
  }
};

// const_binary_op
template <class Op, class E1, class E2>
struct const_binary_op : const_expr_base<const_binary_op<Op, E1, E2>> {
  Op op;
  E1 e1;
  E2 e2;
  template <class OpT, class T1, class T2>
  constexpr const_binary_op(OpT &&op, T1 &&e1, T2 &&e2)
      : op(forward<OpT>(op)), e1(forward<T1>(e1)), e2(forward<T2>(e2)) {}
  template <class... ArgTs> constexpr auto operator()(ArgTs &&... args) const {
    return op(e1(forward<ArgTs>(args)...), e2(forward<ArgTs>(args)...));
  }
  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(op, e1, e2);
  }
};

// overload operators
template <class Op> struct overloaded<Op, category_const_expr> {
  constexpr overloaded() {}
  template <class TT> constexpr decltype(auto) operator()(TT &&v) const {
    return const_unary_op<Op, TT>(Op(), forward<TT>(v));
  }
};

template <class Op>
struct overloaded<Op, category_const_expr, category_const_expr> {
  constexpr overloaded() {}
  template <class TT1, class TT2>
  constexpr decltype(auto) operator()(TT1 &&v1, TT2 &&v2) const {
    return const_binary_op<Op, std::decay_t<TT1>, std::decay_t<TT2>>(
        Op(), forward<TT1>(v1), forward<TT2>(v2));
  }
};

template <class Op> struct overloaded<Op, category_const_expr, void> {
  constexpr overloaded() {}
  template <class TT1, class TT2>
  constexpr decltype(auto) operator()(TT1 &&v1, TT2 &&v2) const {
    return const_binary_op<Op, std::decay_t<TT1>,
                           const_coeff<std::decay_t<TT2>>>(
        Op(), forward<TT1>(v1), as_const_coeff(forward<TT2>(v2)));
  }
};

template <class Op> struct overloaded<Op, void, category_const_expr> {
  constexpr overloaded() {}
  template <class TT1, class TT2>
  constexpr decltype(auto) operator()(TT1 &&v1, TT2 &&v2) const {
    return const_binary_op<Op, const_coeff<std::decay_t<TT1>>,
                           std::decay_t<TT2>>(
        Op(), as_const_coeff(forward<TT1>(v1)), forward<TT2>(v2));
  }
};

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
  return expr(forward<ArgTs>(args)...);
}
template <class T, class = std::enable_if_t<!is_const_expr<T>::value>, class TT,
          class... ArgTs>
constexpr TT &&_pass_or_evaluate(const T &, TT &&expr, ArgTs &&...) {
  return static_cast<TT &&>(expr);
}

// _smart_invoke
template <class FunT, class... ArgTs>
struct _expr_wrapper : const_expr_base<_expr_wrapper<FunT, ArgTs...>> {
  FunT _fun;
  std::tuple<ArgTs...> _args;
  constexpr explicit _expr_wrapper(FunT &&f, ArgTs &&... args)
      : _fun(forward<FunT>(f)), _args(forward<ArgTs>(args)...) {}
  template <class... NewArgTs>
  constexpr decltype(auto) operator()(NewArgTs &&... nargs) const {
    return _invoke_seq(make_const_sequence_for<ArgTs...>(),
                       std::forward<NewArgTs>(nargs)...);
  }
  template <size_t... Is, class... NewArgTs>
  constexpr decltype(auto) _invoke_seq(const const_ints<size_t, Is...> &,
                                       NewArgTs &&... nargs) const {
    return _fun(_pass_or_evaluate(std::get<Is>(_args), std::get<Is>(_args),
                                  nargs...)...);
  }

  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(_fun, _args);
  }
};

template <class FunT, class... ArgTs>
constexpr decltype(auto) _smart_invoke(FunT &&fun, yes there_are_const_exprs,
                                       ArgTs &&... args) {
  return _expr_wrapper<FunT, ArgTs...>(forward<FunT>(fun),
                                       forward<ArgTs>(args)...);
}
template <class FunT, class... ArgTs>
constexpr decltype(auto) _smart_invoke(FunT &&fun, no there_are_const_exprs,
                                       ArgTs &&... args) {
  return fun(forward<ArgTs>(args)...);
}
}

// smart_invoke
// migrate const_exprs into arguments
template <class FunT, class... ArgTs>
constexpr decltype(auto) smart_invoke(FunT &&fun, ArgTs &&... args) {
  return details::_smart_invoke(forward<FunT>(fun),
                                details::_has_const_expr(args...),
                                forward<ArgTs>(args)...);
}
}
