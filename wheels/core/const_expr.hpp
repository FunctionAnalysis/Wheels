#pragma once

#include "overloads.hpp"

namespace wheels {

// const_symbol
template <size_t Idx> struct const_symbol {
  constexpr const_symbol() {}
  template <class... ArgTs> constexpr auto operator()(ArgTs &&... args) const {
    return std::get<Idx>(std::forward_as_tuple(forward<ArgTs>(args)...));
  }
};

template <size_t Idx, class OpT>
constexpr auto category_for_overloading(const const_symbol<Idx> &,
                                        const OpT &) {
  return category_const_expr();
};

namespace literals {
// ""_symbol
template <char... Cs> constexpr auto operator"" _symbol() {
  return const_symbol<details::_parse_int<size_t, Cs...>::value>();
}
}

// const_coeff
template <class T> struct const_coeff {
  T val;
  template <class TT> constexpr const_coeff(TT &&v) : val(forward<TT>(v)) {}
  template <class... ArgTs> constexpr T operator()(ArgTs &&...) const {
    return val;
  }
  template <class V> decltype(auto) fields(V &&visitor) { return visitor(val); }
  template <class V> decltype(auto) fields(V &&visitor) const {
    return visitor(val);
  }
};

template <class T, class Op>
constexpr auto category_for_overloading(const const_coeff<T> &, const Op &) {
  return category_const_expr();
};

template <class T>
constexpr const_coeff<std::decay_t<T>> as_const_coeff(T &&v) {
  return const_coeff<std::decay_t<T>>(forward<T>(v));
}

// const_unary_op
template <class Op, class E> struct const_unary_op {
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
  template <class V> decltype(auto) fields(V &&visitor) const {
    return visitor(op, e);
  }
};

template <class Op, class E, class OtherOp>
constexpr auto category_for_overloading(const const_unary_op<Op, E> &,
                                        const OtherOp &) {
  return category_const_expr();
};

// const_binary_op
template <class Op, class E1, class E2> struct const_binary_op {
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
  template <class V> decltype(auto) fields(V &&visitor) const {
    return visitor(op, e1, e2);
  }
};

template <class Op, class E1, class E2, class OtherOp>
constexpr auto category_for_overloading(const const_binary_op<Op, E1, E2> &,
                                        const OtherOp &) {
  return category_const_expr();
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
    return const_binary_op<Op, TT1, TT2>(Op(), forward<TT1>(v1),
                                         forward<TT2>(v2));
  }
};

template <class Op> struct overloaded<Op, category_const_expr, void> {
  constexpr overloaded() {}
  template <class TT1, class TT2>
  constexpr decltype(auto) operator()(TT1 &&v1, TT2 &&v2) const {
    return const_binary_op<Op, TT1, const_coeff<std::decay_t<TT2>>>(
        Op(), forward<TT1>(v1), as_const_coeff(forward<TT2>(v2)));
  }
};

template <class Op> struct overloaded<Op, void, category_const_expr> {
  constexpr overloaded() {}
  template <class TT1, class TT2>
  constexpr decltype(auto) operator()(TT1 &&v1, TT2 &&v2) const {
    return const_binary_op<Op, const_coeff<std::decay_t<TT1>>, TT2>(
        Op(), as_const_coeff(forward<TT1>(v1)), forward<TT2>(v2));
  }
};
}
