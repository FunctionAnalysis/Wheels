#pragma once

#include "object_fwd.hpp"
#include "overloads.hpp"

#include "const_expr_fwd.hpp"

namespace wheels {

// const_expr_base
template <class T> struct const_expr_base : category::object<T> {};

// is_const_expr
template <class T>
struct is_const_expr : std::is_base_of<const_expr_base<T>, T> {};

// const_symbol
template <size_t Idx> struct const_symbol : const_expr_base<const_symbol<Idx>> {
  constexpr const_symbol() {}
  template <class... ArgTs> constexpr auto operator()(ArgTs &&... args) const {
    return std::get<Idx>(std::forward_as_tuple(std::forward<ArgTs>(args)...));
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
  template <class TT>
  constexpr const_coeff(TT &&v) : val(std::forward<TT>(v)) {}
  template <class... ArgTs> constexpr T operator()(ArgTs &&...) const {
    return val;
  }
  template <class V> decltype(auto) fields(V &&visitor) { return visitor(val); }
};

template <class T>
constexpr const_coeff<std::decay_t<T>> as_const_coeff(T &&v) {
  return const_coeff<std::decay_t<T>>(std::forward<T>(v));
}

// const_unary_op
template <class Op, class E>
struct const_unary_op : const_expr_base<const_unary_op<Op, E>> {
  Op op;
  E e;
  template <class OpT, class T>
  constexpr const_unary_op(OpT &&op, T &&e)
      : op(std::forward<OpT>(op)), e(std::forward<T>(e)) {}
  template <class... ArgTs> constexpr auto operator()(ArgTs &&... args) const {
    return op(e(std::forward<ArgTs>(args)...));
  }
  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(op, e);
  }
};

template <class Op, class E>
constexpr const_unary_op<Op, E> make_unary_op_expr(const Op &op, E &&e) {
  return const_unary_op<Op, E>(op, std::forward<E>(e));
}

// const_binary_op
template <class Op, class E1, class E2>
struct const_binary_op : const_expr_base<const_binary_op<Op, E1, E2>> {
  Op op;
  E1 e1;
  E2 e2;
  template <class OpT, class T1, class T2>
  constexpr const_binary_op(OpT &&op, T1 &&e1, T2 &&e2)
      : op(std::forward<OpT>(op)), e1(std::forward<T1>(e1)),
        e2(std::forward<T2>(e2)) {}
  template <class... ArgTs> constexpr auto operator()(ArgTs &&... args) const {
    return op(e1(std::forward<ArgTs>(args)...),
              e2(std::forward<ArgTs>(args)...));
  }
  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(op, e1, e2);
  }
};

template <class Op, class E1, class E2>
constexpr const_binary_op<Op, E1, E2> make_binary_op_expr(const Op &op, E1 &&e1,
                                                          E2 &&e2) {
  return const_binary_op<Op, E1, E2>(op, std::forward<E1>(e1),
                                     std::forward<E2>(e2));
}

// overload operators
template <class OpT, class T>
constexpr auto overload_as(const func_base<OpT> &, const const_expr_base<T> &) {
  return [](auto &&v) { return make_unary_op_expr(OpT(), wheels_forward(v)); };
}
template <class OpT, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &, const const_expr_base<T1> &,
                           const const_expr_base<T2> &) {
  return [](auto &&v1, auto &&v2) {
    return make_binary_op_expr(OpT(), wheels_forward(v1), wheels_forward(v2));
  };
}

template <class OpT, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &, const const_expr_base<T1> &,
                           const category::other<T2> &) {
  return [](auto &&v1, auto &&v2) {
    return make_binary_op_expr(OpT(), wheels_forward(v1),
                               as_const_coeff(wheels_forward(v2)));
  };
}
template <class OpT, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &, const category::other<T1> &,
                           const const_expr_base<T2> &) {
  return [](auto &&v1, auto &&v2) {
    return make_binary_op_expr(OpT(), as_const_coeff(wheels_forward(v1)),
                               wheels_forward(v2));
  };
}
}
