#pragma once

#include "const_expr_fwd.hpp"
#include "const_ints_fwd.hpp"
#include "object_fwd.hpp"
#include "overloads_fwd.hpp"

#include "utility.hpp"
#include "const_ints.hpp"
#include "overloads.hpp"

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
  constexpr const_coeff(T &&v) : val(std::forward<T>(v)) {}
  template <class... ArgTs> constexpr const T &operator()(ArgTs &&...) const {
    return val;
  }
  template <class V> decltype(auto) fields(V &&visitor) { return visitor(val); }
};

namespace details {
template <class TT, class T>
constexpr TT &&_as_const_coeff_impl(TT &&v, const const_expr_base<T> &) {
  return static_cast<TT &&>(v);
}
template <class TT, class T>
constexpr const_coeff<T> _as_const_coeff_impl(TT &&v,
                                              const category::other<T> &) {
  return const_coeff<T>(std::forward<TT>(v));
}
}
template <class T>
constexpr decltype(auto) as_const_coeff(T && v) {
  return details::_as_const_coeff_impl(std::forward<T>(v),
                                       category::identify(v));
}


// const_unary_op
template <class Op, class E>
struct const_unary_op : const_expr_base<const_unary_op<Op, E>> {
  Op op;
  E e;
  template <class OpT, class T>
  constexpr const_unary_op(OpT &&op, T &&e)
      : op(std::forward<OpT>(op)), e(std::forward<T>(e)) {}
  template <class... ArgTs>
  constexpr decltype(auto) operator()(ArgTs &&... args) const {
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
  template <class... ArgTs>
  constexpr decltype(auto) operator()(ArgTs &&... args) const {
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

// const_call_list
template <class FunT, class... RecordedExprArgTs>
struct const_call_list
    : const_expr_base<const_call_list<FunT, RecordedExprArgTs...>> {
  FunT functor;
  std::tuple<RecordedExprArgTs...> eargs;
 /* static_assert(::wheels::all(is_const_expr<std::decay_t<RecordedExprArgTs>>::value...),
                "All RecordedExprArgTs must be const_expr types");*/

  constexpr explicit const_call_list(FunT f, RecordedExprArgTs &&... as)
      : functor(f), eargs(std::forward<RecordedExprArgTs>(as)...) {}
  template <class... ArgTs>
  constexpr decltype(auto) operator()(ArgTs &&... as) const {
    return const_cast<const_call_list &>(*this)
        ._call_seq(make_const_sequence_for<RecordedExprArgTs...>(),
                   std::forward<ArgTs>(as)...);
  }

private:
  template <size_t... Is, class... ArgTs>
  decltype(auto) _call_seq(const const_ints<size_t, Is...> &, ArgTs &&... as) {
    return functor(std::forward<RecordedExprArgTs>(std::get<Is>(eargs))(
        std::forward<ArgTs>(as)...)...);
  }
};

template <class FunT, class... RecordedExprArgTs>
constexpr auto make_const_call_list(FunT f, RecordedExprArgTs &&... as) {
  return const_call_list<FunT, RecordedExprArgTs...>(
      f, std::forward<RecordedExprArgTs>(as)...);
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


// has_const_expr
namespace details {
template <class T, class... ArgTs>
constexpr yes _has_const_expr_impl(const const_expr_base<T> &,
                                   const ArgTs &...);
template <class T, class... ArgTs>
constexpr auto _has_const_expr_impl(const category::other<T> &,
                                    const ArgTs &... args);
constexpr no _has_const_expr_impl();

template <class T, class... ArgTs>
constexpr yes _has_const_expr_impl(const const_expr_base<T> &,
                                   const ArgTs &...) {
  return yes();
}
template <class T, class... ArgTs>
constexpr auto _has_const_expr_impl(const category::other<T> &,
                                    const ArgTs &... args) {
  return _has_const_expr_impl(args...);
}
constexpr no _has_const_expr_impl() { return no(); }
}
template <class... ArgTs> constexpr auto has_const_expr(const ArgTs &... args) {
  return details::_has_const_expr_impl(category::identify(args)...);
}
}
