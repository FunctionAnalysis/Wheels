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

#include "ewise_fwd.hpp"

#include "overloads.hpp"
#include "what.hpp"

#include "extension.hpp"
#include "tensor_base.hpp"
#include "types.hpp"

namespace wheels {

// ewise_op_result
template <class EleT, class ShapeT, class OpT, class InputT, class... InputTs>
class ewise_op_result
    : public ewise_base<EleT, ShapeT, ewise_op_result<EleT, ShapeT, OpT, InputT,
                                                      InputTs...>> {
public:
  using shape_type = ShapeT;
  using value_type = EleT;
  constexpr explicit ewise_op_result(OpT o, InputT &&in, InputTs &&... ins)
      : op(o), inputs(std::forward<InputT>(in), std::forward<InputTs>(ins)...) {
  }

public:
  OpT op;
  std::tuple<InputT, InputTs...> inputs;
};

// make_ewise_op_result
template <class ET, class ShapeT, class OpT, class InputT, class... InputTs>
constexpr ewise_op_result<ET, ShapeT, OpT, InputT, InputTs...>
make_ewise_op_result(OpT op, InputT &&input, InputTs &&... inputs) {
  return ewise_op_result<ET, ShapeT, OpT, InputT, InputTs...>(
      op, std::forward<InputT>(input), std::forward<InputTs>(inputs)...);
}

// shape_of
template <class EleT, class ShapeT, class OpT, class InputT, class... InputTs>
constexpr decltype(auto)
shape_of(const ewise_op_result<EleT, ShapeT, OpT, InputT, InputTs...> &ts) {
  return shape_of(std::get<0>(ts.inputs));
}

// element_at
namespace detail {
template <class EwiseOpResultT, size_t... Is, class... SubTs>
constexpr decltype(auto)
_element_at_ewise_op_result_seq(EwiseOpResultT &&ts,
                                const const_ints<size_t, Is...> &,
                                const SubTs &... subs) {
  return ts.op(element_at(std::get<Is>(ts.inputs), subs...)...);
}
}
template <class EleT, class ShapeT, class OpT, class InputT, class... InputTs,
          class... SubTs>
constexpr decltype(auto)
element_at(const ewise_op_result<EleT, ShapeT, OpT, InputT, InputTs...> &ts,
           const SubTs &... subs) {
  assert(subscripts_are_valid(ts.shape(), subs...));
  return detail::_element_at_ewise_op_result_seq(
      ts, make_const_sequence_for<InputT, InputTs...>(), subs...);
}

// element_at_index
namespace detail {
template <class EwiseOpResultT, size_t... Is, class IndexT>
constexpr decltype(auto)
_element_at_index_ewise_op_result_seq(EwiseOpResultT &&ts,
                                      const const_ints<size_t, Is...> &,
                                      const IndexT &index) {
  return ts.op(element_at_index(std::get<Is>(ts.inputs), index)...);
}
}
template <class EleT, class ShapeT, class OpT, class InputT, class... InputTs,
          class IndexT>
constexpr decltype(auto) element_at_index(
    const ewise_op_result<EleT, ShapeT, OpT, InputT, InputTs...> &ts,
    const IndexT &index) {
  assert(is_between(index, 0, (typename int_traits<IndexT>::type)ts.numel()));
  return detail::_element_at_index_ewise_op_result_seq(
      ts, make_const_sequence_for<InputT, InputTs...>(), index);
}

// for_each_element
namespace details {
template <class FunT, class EwiseOpResultT, class... InputTs,
          class AllElesTupleT, size_t... Is, size_t... Js>
constexpr decltype(auto) _for_element_in_ewise_op_result_helper(
    FunT fun, const EwiseOpResultT &t, AllElesTupleT &&all_eles,
    const const_ints<size_t, Is...> &, const const_ints<size_t, Js...> &) {
  return fun(t.op(std::get<Is>(all_eles)...),
             std::get<Js + sizeof...(Is)>(all_eles)...);
}

template <class FunT, class EleT, class ShapeT, class OpT, class InputT,
          class... InputTs, class... AllEleTs>
constexpr decltype(auto) _for_element_in_ewise_op_result(
    FunT fun, const ewise_op_result<EleT, ShapeT, OpT, InputT, InputTs...> &t,
    AllEleTs &&... all_eles) {
  return _for_element_in_ewise_op_result_helper(
      fun, t, std::forward_as_tuple(std::forward<AllEleTs>(all_eles)...),
      make_const_sequence_for<InputT, InputTs...>(),
      make_const_sequence(
          const_size<sizeof...(AllEleTs)-1 - sizeof...(InputTs)>()));
}

template <behavior_flag_enum F, class FunT, class EleT, class ShapeT, class OpT,
          class InputT, class... InputTs, size_t... Is, class... Ts>
constexpr bool _for_each_element_in_ewise_op_result_seq(
    behavior_flag<F> f, FunT fun,
    const ewise_op_result<EleT, ShapeT, OpT, InputT, InputTs...> &t,
    const const_ints<size_t, Is...> &, Ts &&... ts) {
  return for_each_element(f,
                          [fun, &t](auto &&... es) {
                            return _for_element_in_ewise_op_result(
                                fun, t, wheels_forward(es)...);
                          },
                          std::get<Is>(t.inputs)..., std::forward<Ts>(ts)...);
}
}
template <behavior_flag_enum F, class FunT, class EleT, class ShapeT, class OpT,
          class InputT, class... InputTs, class... Ts>
constexpr bool for_each_element(
    behavior_flag<F> f, FunT fun,
    const ewise_op_result<EleT, ShapeT, OpT, InputT, InputTs...> &t,
    Ts &&... ts) {
  return details::_for_each_element_in_ewise_op_result_seq(
      f, fun, t, make_const_sequence_for<InputT, InputTs...>(),
      std::forward<Ts>(ts)...);
}

// most ewise binray ops apply on two tensors (except certain ops like below)
template <class OpT, class, class EleT, class ShapeT, class T, class... EleTs,
          class... ShapeTs, class... Ts>
constexpr auto overload_as(const func_base<OpT> &op,
                           const tensor_base<EleT, ShapeT, T> &t,
                           const tensor_base<EleTs, ShapeTs, Ts> &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  using ele_t = std::decay_t<decltype(
      eval(OpT()(std::declval<EleT>(), std::declval<EleTs>()...)))>;
  return [](auto &&t, auto &&... ts) {
    return make_ewise_op_result<ele_t, ShapeT>(OpT(), wheels_forward(t),
                                               wheels_forward(ts)...);
  };
}

// all ewise binary ops apply on ewised tensor
// t1.ewise() == t2, t1.ewise() * t2
template <class OpT, class EleT, class ShapeT, class T, class... EleTs,
          class... ShapeTs, class... Ts>
constexpr auto overload_as(const func_base<OpT> &op,
                           const ewise_wrapper<EleT, ShapeT, T> &t,
                           const tensor_base<EleTs, ShapeTs, Ts> &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  using ele_t = std::decay_t<decltype(
      eval(OpT()(std::declval<EleT>(), std::declval<EleTs>()...)))>;
  return [](auto &&t, auto &&... ts) {
    return make_ewise_op_result<ele_t, ShapeT>(OpT(), wheels_forward(t).host,
                                               wheels_forward(ts)...);
  };
}

// tensor vs scalar
template <class OpT, class EleT1, class ShapeT1, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const tensor_base<EleT1, ShapeT1, T1> &,
                           const proxy_base<T2> &) {
  using ele_t = std::decay_t<decltype(
      eval(OpT()(std::declval<EleT1>(), std::declval<T2>())))>;
  return [](auto &&t1, auto &&t2) {
    return make_ewise_op_result<ele_t, ShapeT1>(
        OpT()(const_arg<0>(), wheels_forward(t2)), wheels_forward(t1));
  };
}
template <class OpT, class EleT1, class ShapeT1, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const tensor_base<EleT1, ShapeT1, T1> &,
                           const scalar<T2> &) {
  using ele_t = std::decay_t<decltype(
      eval(OpT()(std::declval<EleT1>(), std::declval<T2>())))>;
  return [](auto &&t1, auto t2) {
    return make_ewise_op_result<ele_t, ShapeT1>(
        OpT()(const_arg<0>(), std::move(t2)), wheels_forward(t1));
  };
}

template <class OpT, class EleT1, class ShapeT1, class EleT2, class ShapeT2,
          class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const tensor_base<EleT1, ShapeT1, T1> &,
                           const scalarize_wrapper<EleT2, ShapeT2, T2> &) {
  using ele_t = std::decay_t<decltype(
      eval(OpT()(std::declval<EleT1>(), std::declval<T2>())))>;
  return [](auto &&t1, auto &&t2) {
    return make_ewise_op_result<ele_t, ShapeT1>(
        OpT()(const_arg<0>(), wheels_forward(t2).host), wheels_forward(t1));
  };
}

// scalar vs tensor
template <class OpT, class T1, class EleT2, class ShapeT2, class T2>
constexpr auto overload_as(const func_base<OpT> &op, const proxy_base<T1> &,
                           const tensor_base<EleT2, ShapeT2, T2> &) {
  using ele_t = std::decay_t<decltype(
      eval(OpT()(std::declval<T1>(), std::declval<EleT2>())))>;
  return [](auto &&t1, auto &&t2) {
    return make_ewise_op_result<ele_t, ShapeT2>(
        OpT()(wheels_forward(t1), const_arg<0>()), wheels_forward(t2));
  };
}
template <class OpT, class T1, class EleT2, class ShapeT2, class T2>
constexpr auto overload_as(const func_base<OpT> &op, const scalar<T1> &,
                           const tensor_base<EleT2, ShapeT2, T2> &) {
  using ele_t = std::decay_t<decltype(
      eval(OpT()(std::declval<T1>(), std::declval<EleT2>())))>;
  return [](auto t1, auto &&t2) {
    return make_ewise_op_result<ele_t, ShapeT2>(
        OpT()(std::move(t1), const_arg<0>()), wheels_forward(t2));
  };
}

template <class OpT, class EleT1, class ShapeT1, class EleT2, class ShapeT2,
          class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const scalarize_wrapper<EleT1, ShapeT1, T1> &,
                           const tensor_base<EleT2, ShapeT2, T2> &) {
  using ele_t = std::decay_t<decltype(
      eval(OpT()(std::declval<T1>(), std::declval<EleT2>())))>;
  return [](auto &&t1, auto &&t2) {
    return make_ewise_op_result<ele_t, ShapeT2>(
        OpT()(wheels_forward(t1).host, const_arg<0>()), wheels_forward(t2));
  };
}

// tensor vs const_expr
template <class OpT, class EleT1, class ShapeT1, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const tensor_base<EleT1, ShapeT1, T1> &,
                           const const_expr_base<T2> &) {
  return [](auto &&t1, auto &&t2) {
    return make_const_call_list(OpT(), as_const_coeff(wheels_forward(t1)),
                                wheels_forward(t2));
  };
}

// const_expr vs tensor
template <class OpT, class T1, class EleT2, class ShapeT2, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const const_expr_base<T1> &,
                           const tensor_base<EleT2, ShapeT2, T2> &) {
  return [](auto &&t1, auto &&t2) {
    return make_const_call_list(OpT(), wheels_forward(t1),
                                as_const_coeff(wheels_forward(t2)));
  };
}

namespace detail {
// auto transform(ts)
template <class EleT, class ShapeT, class T, class FunT>
constexpr auto _transform(const tensor_base<EleT, ShapeT, T> &t, FunT &&fun) {
  using ele_t = std::decay_t<decltype(eval(fun(std::declval<EleT>())))>;
  return make_ewise_op_result<ele_t, ShapeT>(std::forward<FunT>(fun),
                                             t.derived());
}
template <class EleT, class ShapeT, class T, class FunT>
constexpr auto _transform(tensor_base<EleT, ShapeT, T> &&t, FunT &&fun) {
  using ele_t = std::decay_t<decltype(eval(fun(std::declval<EleT>())))>;
  return make_ewise_op_result<ele_t, ShapeT>(std::forward<FunT>(fun),
                                             std::move(t.derived()));
}
}

// equals & not_equals
namespace detail {

// with tensor
template <class EleT1, class ShapeT1, class T1, class T2, class TT1, class TT2>
constexpr auto _ewise_equals(const tensor_base<EleT1, ShapeT1, T1> &,
                             const tensor_core<T2> &, TT1 &&t1, TT2 &&t2) {
  assert(t1.shape() == t2.shape());
  return make_ewise_op_result<bool, ShapeT1>(
      binary_op_eq(), std::forward<TT1>(t1), std::forward<TT2>(t2));
}

// with scalar
template <class EleT1, class ShapeT1, class T1, class T2, class TT1, class TT2>
constexpr auto _ewise_equals(const tensor_base<EleT1, ShapeT1, T1> &,
                             const proxy_base<T2> &, TT1 &&t1, TT2 &&t2) {
  return make_ewise_op_result<bool, ShapeT1>(
      binary_op_eq()(const_arg<0>(), std::forward<TT2>(t2)),
      std::forward<TT1>(t1));
}

// with tensor
template <class EleT1, class ShapeT1, class T1, class T2, class TT1, class TT2>
constexpr auto _ewise_not_equals(const tensor_base<EleT1, ShapeT1, T1> &,
                                 const tensor_core<T2> &, TT1 &&t1, TT2 &&t2) {
  assert(t1.shape() == t2.shape());
  return make_ewise_op_result<bool, ShapeT1>(
      binary_op_neq(), std::forward<TT1>(t1), std::forward<TT2>(t2));
}

// with scalar
template <class EleT1, class ShapeT1, class T1, class T2, class TT1, class TT2>
constexpr auto _ewise_not_equals(const tensor_base<EleT1, ShapeT1, T1> &,
                                 const proxy_base<T2> &, TT1 &&t1, TT2 &&t2) {
  return make_ewise_op_result<bool, ShapeT1>(
      binary_op_neq()(const_arg<0>(), std::forward<TT2>(t2)),
      std::forward<TT1>(t1));
}
}

template <class EleT, class ShapeT, class T>
template <class AnotherT>
inline constexpr auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::equals(
    AnotherT &&t) const & {
  return detail::_ewise_equals(this->derived(), what(t), this->derived(),
                                std::forward<AnotherT>(t));
}

template <class EleT, class ShapeT, class T>
template <class AnotherT>
inline auto tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::equals(
    AnotherT &&t) & {
  return detail::_ewise_equals(this->derived(), what(t), this->derived(),
                                std::forward<AnotherT>(t));
}

template <class EleT, class ShapeT, class T>
template <class AnotherT>
inline auto tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::equals(
    AnotherT &&t) && {
  return detail::_ewise_equals(this->derived(), what(t),
                                std::move(this->derived()),
                                std::forward<AnotherT>(t));
}

template <class EleT, class ShapeT, class T>
template <class AnotherT>
inline constexpr auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::not_equals(
    AnotherT &&t) const & {
  return detail::_ewise_not_equals(this->derived(), what(t), this->derived(),
                                    std::forward<AnotherT>(t));
}

template <class EleT, class ShapeT, class T>
template <class AnotherT>
inline auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::not_equals(
    AnotherT &&t) & {
  return detail::_ewise_not_equals(this->derived(), what(t), this->derived(),
                                    std::forward<AnotherT>(t));
}

template <class EleT, class ShapeT, class T>
template <class AnotherT>
inline auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::not_equals(
    AnotherT &&t) && {
  return detail::_ewise_not_equals(this->derived(), what(t),
                                    std::move(this->derived()),
                                    std::forward<AnotherT>(t));
}

// transform
template <class EleT, class ShapeT, class T>
template <class FunT>
inline constexpr auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::transform(
    FunT &&fun) const & {
  return detail::_transform(this->derived(), std::forward<FunT>(fun));
}
template <class EleT, class ShapeT, class T>
template <class FunT>
inline auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::transform(
    FunT &&fun) & {
  return detail::_transform(this->derived(), std::forward<FunT>(fun));
}
template <class EleT, class ShapeT, class T>
template <class FunT>
inline auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::transform(
    FunT &&fun) && {
  return detail::_transform(std::move(this->derived()),
                             std::forward<FunT>(fun));
}

// cast
template <class EleT, class ShapeT, class T>
template <cast_type_enum cast_type, class TargetEleT>
inline constexpr auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::cast() const & {
  return detail::_transform(this->derived(), [](const auto &e) -> TargetEleT {
    return ::wheels::cast<cast_type, TargetEleT>(e);
  });
}
template <class EleT, class ShapeT, class T>
template <cast_type_enum cast_type, class TargetEleT>
inline auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::cast() && {
  return detail::_transform(std::move(this->derived()),
                             [](auto &&e) -> TargetEleT {
                               return ::wheels::cast<cast_type, TargetEleT>(e);
                             });
}

// _as_tuple_seq
namespace detail {
template <class FirstTT, class... TTs, size_t... Is, class FirstEleT,
          class... EleTs, class FirstShapeT, class... ShapeTs, class FirstT,
          class... Ts>
constexpr auto _as_tuple_seq(std::tuple<FirstTT, TTs...> &&ts,
                             const const_ints<size_t, Is...> &,
                             const ewise_base<FirstEleT, FirstShapeT, FirstT> &,
                             const ewise_base<EleTs, ShapeTs, Ts> &...) {
  using result_ele_t = std::tuple<FirstEleT, EleTs...>;
  return make_ewise_op_result<result_ele_t, FirstShapeT>(
      [](auto &&... es) { return as_tuple(wheels_forward(es)...); },
      std::get<0>(ts), std::get<Is + 1>(ts)...);
}
}
}