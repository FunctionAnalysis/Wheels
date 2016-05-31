#pragma once

#include "ewise_fwd.hpp"
#include "object_fwd.hpp"

#include "object.hpp"
#include "overloads.hpp"

#include "extension.hpp"
#include "tensor_base.hpp"

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
namespace details {
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
  return details::_element_at_ewise_op_result_seq(
      ts, make_const_sequence_for<InputT, InputTs...>(), subs...);
}

// element_at_index
namespace details {
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
  return details::_element_at_index_ewise_op_result_seq(
      ts, make_const_sequence_for<InputT, InputTs...>(), index);
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
                           const category::other<T2> &) {
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
                           const category::scalar<T2> &) {
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
constexpr auto overload_as(const func_base<OpT> &op,
                           const category::other<T1> &,
                           const tensor_base<EleT2, ShapeT2, T2> &) {
  using ele_t = std::decay_t<decltype(
      eval(OpT()(std::declval<T1>(), std::declval<EleT2>())))>;
  return [](auto &&t1, auto &&t2) {
    return make_ewise_op_result<ele_t, ShapeT2>(
        OpT()(wheels_forward(t1), const_arg<0>()), wheels_forward(t2));
  };
}
template <class OpT, class T1, class EleT2, class ShapeT2, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const category::scalar<T1> &,
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

namespace details {
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

// cast
template <class TargetEleT, class EleT, class ShapeT, class T>
constexpr auto _static_ecast(const tensor_base<EleT, ShapeT, T> &t) {
  return _transform(t.derived(),
                    [](const auto &e) { return static_cast<TargetEleT>(e); });
}
template <class TargetEleT, class EleT, class ShapeT, class T>
constexpr auto _static_ecast(tensor_base<EleT, ShapeT, T> &&t) {
  return _transform(std::move(t.derived()),
                    [](const auto &e) { return static_cast<TargetEleT>(e); });
}
}

// equals & not_equals
namespace details {

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
                             const category::other<T2> &, TT1 &&t1, TT2 &&t2) {
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
                                 const category::other<T2> &, TT1 &&t1,
                                 TT2 &&t2) {
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
  return details::_ewise_equals(this->derived(), category::identify(t),
                                this->derived(), std::forward<AnotherT>(t));
}

template <class EleT, class ShapeT, class T>
template <class AnotherT>
inline auto tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::equals(
    AnotherT &&t) & {
  return details::_ewise_equals(this->derived(), category::identify(t),
                                this->derived(), std::forward<AnotherT>(t));
}

template <class EleT, class ShapeT, class T>
template <class AnotherT>
inline auto tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::equals(
    AnotherT &&t) && {
  return details::_ewise_equals(this->derived(), category::identify(t),
                                std::move(this->derived()),
                                std::forward<AnotherT>(t));
}

template <class EleT, class ShapeT, class T>
template <class AnotherT>
inline constexpr auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::not_equals(
    AnotherT &&t) const & {
  return details::_ewise_not_equals(this->derived(), category::identify(t),
                                    this->derived(), std::forward<AnotherT>(t));
}

template <class EleT, class ShapeT, class T>
template <class AnotherT>
inline auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::not_equals(
    AnotherT &&t) & {
  return details::_ewise_not_equals(this->derived(), category::identify(t),
                                    this->derived(), std::forward<AnotherT>(t));
}

template <class EleT, class ShapeT, class T>
template <class AnotherT>
inline auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::not_equals(
    AnotherT &&t) && {
  return details::_ewise_not_equals(this->derived(), category::identify(t),
                                    std::move(this->derived()),
                                    std::forward<AnotherT>(t));
}

// transform
template <class EleT, class ShapeT, class T>
template <class FunT>
inline constexpr auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::transform(
    FunT &&fun) const & {
  return details::_transform(this->derived(), std::forward<FunT>(fun));
}
template <class EleT, class ShapeT, class T>
template <class FunT>
inline auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::transform(
    FunT &&fun) & {
  return details::_transform(this->derived(), std::forward<FunT>(fun));
}
template <class EleT, class ShapeT, class T>
template <class FunT>
inline auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::transform(
    FunT &&fun) && {
  return details::_transform(std::move(this->derived()),
                             std::forward<FunT>(fun));
}

// cast
template <class EleT, class ShapeT, class T>
template <class TargetEleT>
inline constexpr auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::cast() const & {
  return details::_static_ecast<TargetEleT>(this->derived());
}
template <class EleT, class ShapeT, class T>
template <class TargetEleT>
inline auto
tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>::cast() && {
  return details::_static_ecast<TargetEleT>(std::move(this->derived()));
}

// _as_tuple_seq
namespace details {
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