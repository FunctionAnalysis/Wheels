/* * *
 * The MIT License (MIT)
 * 
 * Copyright (c) 2016 Hao Yang (yangh2007@gmail.com)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * * */

#pragma once

#include "const_expr_fwd.hpp"
#include "const_ints_fwd.hpp"
#include "overloads_fwd.hpp"
#include "what_fwd.hpp"

#include "const_ints.hpp"
#include "overloads.hpp"
#include "utility.hpp"

namespace wheels {
// const_expr_base
template <class T> struct const_expr_base : object_base<T> {
  template <class... ArgTs>
  constexpr decltype(auto) operator()(ArgTs &&... args) const & {
    return invoke_const_expr(this->derived(), std::forward<ArgTs>(args)...);
  }
  template <class... ArgTs> decltype(auto) operator()(ArgTs &&... args) & {
    return invoke_const_expr(this->derived(), std::forward<ArgTs>(args)...);
  }
  template <class... ArgTs> decltype(auto) operator()(ArgTs &&... args) && {
    return invoke_const_expr(std::move(this->derived()),
                             std::forward<ArgTs>(args)...);
  }
};

template <class T, class K>
constexpr T eval_what(const K &v, const const_expr_base<T> &) {
  return T(v);
}

// const_arg
template <size_t Idx> struct const_arg : const_expr_base<const_arg<Idx>> {
  constexpr const_arg() {}
  template <class... ArgTs> constexpr auto operator()(ArgTs &&... args) const {
    return std::get<Idx>(std::forward_as_tuple(std::forward<ArgTs>(args)...));
  }
};

namespace literals {
// ""_arg
template <char... Cs> constexpr auto operator"" _arg() {
  return const_arg<detail::_parse_int<size_t, Cs...>::value>();
}
}

// invoke_const_expr_impl
// used by invoke_const_expr
template <size_t Idx, class EE, class... ArgTs>
decltype(auto) invoke_const_expr_impl(const const_arg<Idx> &, EE &&,
                                      ArgTs &&... args) {
  return std::get<Idx>(std::forward_as_tuple(std::forward<ArgTs>(args)...));
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

// as_const_coeff
// wrap a non const expr type as a const expr
namespace detail {
template <class TT, class T>
constexpr TT &&_as_const_coeff_impl(TT &&v, const const_expr_base<T> &) {
  return static_cast<TT &&>(v);
}
template <class TT, class T>
constexpr const_coeff<TT> _as_const_coeff_impl(TT &&v, const object_base<T> &) {
  return const_coeff<TT>(std::forward<TT>(v));
}
template <class TT, class T>
constexpr const_coeff<TT> _as_const_coeff_impl(TT &&v, const proxy_base<T> &) {
  return const_coeff<TT>(std::forward<TT>(v));
}
}
template <class T> constexpr decltype(auto) as_const_coeff(T &&v) {
  return detail::_as_const_coeff_impl(std::forward<T>(v), what(v));
}

// invoke_const_expr_impl
// used by invoke_const_expr
template <class T, class EE, class... ArgTs>
decltype(auto) invoke_const_expr_impl(const const_coeff<T> &, EE &&e,
                                      ArgTs &&... args) {
  return std::forward<EE>(e).val;
}

// const_call_list
template <class FunT, class... RecordedExprArgTs>
struct const_call_list
    : const_expr_base<const_call_list<FunT, RecordedExprArgTs...>> {
  FunT functor;
  std::tuple<RecordedExprArgTs...> bind_expr_args;

  constexpr explicit const_call_list(FunT &&f, RecordedExprArgTs &&... as)
      : functor(std::forward<FunT>(f)),
        bind_expr_args(std::forward<RecordedExprArgTs>(as)...) {}
};

// make_const_call_list
template <class FunT, class... RecordedExprArgTs>
constexpr auto make_const_call_list(FunT &&f, RecordedExprArgTs &&... as) {
  return const_call_list<FunT, RecordedExprArgTs...>(
      std::forward<FunT>(f), std::forward<RecordedExprArgTs>(as)...);
}

namespace detail {
template <size_t... Is, class EE, class... ArgTs>
decltype(auto) _invoke_const_call_list_seq(const const_ints<size_t, Is...> &,
                                           EE &&e, ArgTs &&... args) {
  return std::forward<EE>(e).functor(
      invoke_const_expr(std::get<Is>(std::forward<EE>(e).bind_expr_args),
                        std::forward<ArgTs>(args)...)...);
}
}

// invoke_const_expr_impl
// used by invoke_const_expr
template <class FunT, class... RecordedExprArgTs, class EE, class... ArgTs>
decltype(auto)
invoke_const_expr_impl(const const_call_list<FunT, RecordedExprArgTs...> &,
                       EE &&e, ArgTs &&... args) {
  return detail::_invoke_const_call_list_seq(
      make_const_sequence_for<RecordedExprArgTs...>(), std::forward<EE>(e),
      std::forward<ArgTs>(args)...);
}

// overload operators
template <class OpT, class T>
constexpr auto overload_as(const func_base<OpT> &, const const_expr_base<T> &) {
  return
      [](auto &&v) { return make_const_call_list(OpT(), wheels_forward(v)); };
}
template <class OpT, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &, const const_expr_base<T1> &,
                           const const_expr_base<T2> &) {
  return [](auto &&v1, auto &&v2) {
    return make_const_call_list(OpT(), wheels_forward(v1), wheels_forward(v2));
  };
}

template <class OpT, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &, const const_expr_base<T1> &,
                           const proxy_base<T2> &) {
  return [](auto &&v1, auto &&v2) {
    return make_const_call_list(OpT(), wheels_forward(v1),
                                as_const_coeff(wheels_forward(v2)));
  };
}
template <class OpT, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &, const proxy_base<T1> &,
                           const const_expr_base<T2> &) {
  return [](auto &&v1, auto &&v2) {
    return make_const_call_list(OpT(), as_const_coeff(wheels_forward(v1)),
                                wheels_forward(v2));
  };
}

// invoke_const_expr
template <class E, class... ArgTs>
decltype(auto) invoke_const_expr(E &&e, ArgTs &&... args) {
  return invoke_const_expr_impl(e, std::forward<E>(e),
                                std::forward<ArgTs>(args)...);
}

// has_const_expr
namespace detail {
template <class T, class... ArgTs>
constexpr yes _has_const_expr_impl(const const_expr_base<T> &,
                                   const ArgTs &...);
template <class T, class... ArgTs>
constexpr auto _has_const_expr_impl(const proxy_base<T> &,
                                    const ArgTs &... args);
constexpr no _has_const_expr_impl();

template <class T, class... ArgTs>
constexpr yes _has_const_expr_impl(const const_expr_base<T> &,
                                   const ArgTs &...) {
  return yes();
}
template <class T, class... ArgTs>
constexpr auto _has_const_expr_impl(const proxy_base<T> &,
                                    const ArgTs &... args) {
  return _has_const_expr_impl(args...);
}
constexpr no _has_const_expr_impl() { return no(); }
}
template <class... ArgTs> constexpr auto has_const_expr(const ArgTs &... args) {
  return detail::_has_const_expr_impl(what(args)...);
}

namespace detail {
// _smart_invoke
template <class FunT, class... ArgTs>
constexpr auto _smart_invoke(FunT &&fun, yes there_are_const_exprs,
                             ArgTs &&... args) {
  // return _functor_expr<FunT, ArgTs...>(fun, std::forward<ArgTs>(args)...);
  return make_const_call_list(std::forward<FunT>(fun),
                              as_const_coeff(std::forward<ArgTs>(args))...);
}
template <class FunT, class... ArgTs>
constexpr decltype(auto) _smart_invoke(FunT &&fun, no there_are_const_exprs,
                                       ArgTs &&... args) {
  return std::forward<FunT>(fun)(std::forward<ArgTs>(args)...);
}
}

// smart_invoke
// migrate const_exprs into arguments
template <class FunT, class... ArgTs>
constexpr decltype(auto) smart_invoke(FunT &&fun, ArgTs &&... args) {
  return detail::_smart_invoke(std::forward<FunT>(fun),
                                has_const_expr(args...),
                                std::forward<ArgTs>(args)...);
}
}
