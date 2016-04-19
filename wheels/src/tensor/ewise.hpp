#pragma once

#include "ewise_fwd.hpp"

#include "../core/overloads.hpp"

#include "base.hpp"

namespace wheels {

// ewise_wrapper
template <class EleT, class ShapeT, class T>
class ewise_wrapper
    : public ewise_base<EleT, ShapeT, ewise_wrapper<EleT, ShapeT, T>> {
public:
  explicit ewise_wrapper(T &&h) : host(std::forward<T>(h)) {}
  T host;
};

// -- necessary tensor functions
// Shape shape_of(ts);
template <class EleT, class ShapeT, class T>
constexpr decltype(auto) shape_of(const ewise_wrapper<EleT, ShapeT, T> &t) {
  return shape_of(t.host);
}

// Scalar element_at(ts, subs ...);
template <class EleT, class ShapeT, class T, class... SubTs>
constexpr decltype(auto) element_at(const ewise_wrapper<EleT, ShapeT, T> &t,
                                    const SubTs &... subs) {
  return element_at(t.host, subs);
}
template <class EleT, class ShapeT, class T, class... SubTs>
decltype(auto) element_at(ewise_wrapper<EleT, ShapeT, T> &t,
                          const SubTs &... subs) {
  return element_at(t.host, subs);
}

// Scalar element_at_index(ts, index);
template <class EleT, class ShapeT, class T, class IndexT>
constexpr decltype(auto)
element_at_index(const ewise_wrapper<EleT, ShapeT, T> &t, const IndexT &ind) {
  return element_at_index(t.host, ind);
}
template <class EleT, class ShapeT, class T, class IndexT>
decltype(auto) element_at_index(ewise_wrapper<EleT, ShapeT, T> &t,
                                const IndexT &ind) {
  return element_at_index(t.host, ind);
}

// void reserve_shape(ts, shape);
template <class EleT, class ShapeT, class T, class ST, class... SizeTs>
void reserve_shape(ewise_wrapper<EleT, ShapeT, T> &t,
                   const tensor_shape<ST, SizeTs...> &shape) {
  reserve_shape(t.host, shape);
}

// for_each_element
template <behavior_flag_enum F, class FunT, class EleT, class ShapeT, class T,
          class... Ts>
bool for_each_element(behavior_flag<F> f, FunT &fun,
                      const ewise_wrapper<EleT, ShapeT, T> &t, Ts &... ts) {
  return for_each_element(f, fun, t.host, ts...);
}

// fill_elements_with
template <class EleT, class ShapeT, class T, class E>
void fill_elements_with(ewise_wrapper<EleT, ShapeT, T> &t, const E &e) {
  fill_elements_with(t.host, e);
}

// size_t nonzero_elements_count(t)
template <class EleT, class ShapeT, class T>
size_t nonzero_elements_count(const ewise_wrapper<EleT, ShapeT, T> &t) {
  return nonzero_elements_count(t.host);
}

// Scalar reduce_elements(ts, initial, functor);
template <class EleT, class ShapeT, class T, class E, class ReduceT>
E reduce_elements(const ewise_wrapper<EleT, ShapeT, T> &t, E initial,
                  ReduceT &red) {
  return reduce_elements(t.host, initial, red);
}

// Scalar norm_squared(ts)
template <class ET, class ShapeT, class T>
ET norm_squared(const ewise_wrapper<ET, ShapeT, T> &t) {
  return norm_squared(t.host);
}

// Scalar norm(ts)
template <class ET, class ShapeT, class T>
constexpr auto norm(const ewise_wrapper<ET, ShapeT, T> &t) {
  return norm(t.host);
}

// bool all(s)
template <class ET, class ShapeT, class T>
constexpr bool all_of(const ewise_wrapper<ET, ShapeT, T> &t) {
  return all_of(t.host);
}

// bool any(s)
template <class ET, class ShapeT, class T>
constexpr bool any_of(const ewise_wrapper<ET, ShapeT, T> &t) {
  return any_of(t.host);
}

// equals_result_of
template <class ET, class ShapeT, class T>
constexpr bool equals_result_of(const ewise_wrapper<ET, ShapeT, T> &t) {
  return equals_result_of(t.host);
}

// not_equals_result_of
template <class ET, class ShapeT, class T>
constexpr bool not_equals_result_of(const ewise_wrapper<ET, ShapeT, T> &t) {
  return not_equals_result_of(t.host);
}

// Scalar sum(s)
template <class ET, class ShapeT, class T>
ET sum_of(const ewise_wrapper<ET, ShapeT, T> &t) {
  return sum_of(t.host);
}

namespace details {
template <class EleT, class ShapeT, class T, class TT>
constexpr auto _ewise(const tensor_base<EleT, ShapeT, T> &, TT &&host) {
  return ewise_wrapper<EleT, ShapeT, TT>(std::forward<TT>(host));
}
template <class EleT, class ShapeT, class T, class TT>
constexpr TT &&_ewise(const ewise_base<EleT, ShapeT, T> &, TT &&host) {
  return static_cast<TT &&>(host);
}
}

// ewise ops
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
_element_at_ewise_op_result_seq(EwiseOpResultT &ts,
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

// equals_result_of
namespace details {
template <class EwiseOpResultT, size_t... Is>
constexpr bool
_equals_result_of_ewise_op_result_seq(const EwiseOpResultT &ts,
                                      const const_ints<size_t, Is...> &) {
  return all_same(std::get<Is>(ts.inputs).shape()...) && all_of(ts);
}
}
template <class EleT, class ShapeT, class OpT, class InputT, class... InputTs>
constexpr bool equals_result_of(
    const ewise_op_result<EleT, ShapeT, OpT, InputT, InputTs...> &ts) {
  return details::_equals_result_of_ewise_op_result_seq(
      ts, make_const_sequence_for<InputT, InputTs...>());
}

// not_equals_result_of
namespace details {
template <class EwiseOpResultT, size_t... Is>
constexpr bool
_not_equals_result_of_ewise_op_result_seq(const EwiseOpResultT &ts,
                                          const const_ints<size_t, Is...> &) {
  return !all_same(std::get<Is>(ts.inputs).shape()...) || any_of(ts);
}
}
template <class EleT, class ShapeT, class OpT, class InputT, class... InputTs>
constexpr bool not_equals_result_of(
    const ewise_op_result<EleT, ShapeT, OpT, InputT, InputTs...> &ts) {
  return details::_not_equals_result_of_ewise_op_result_seq(
      ts, make_const_sequence_for<InputT, InputTs...>());
}

// element_at_index
namespace details {
template <class EwiseOpResultT, size_t... Is, class IndexT>
constexpr decltype(auto)
_element_at_index_ewise_op_result_seq(EwiseOpResultT &ts,
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

// all tensors
template <class OpT, class EleT, class ShapeT, class T, class... EleTs,
          class... ShapeTs, class... Ts>
constexpr auto overload_as(const func_base<OpT> &op,
                           const ewise_base<EleT, ShapeT, T> &t,
                           const ewise_base<EleTs, ShapeTs, Ts> &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  using ele_t = std::decay_t<decltype(
      OpT()(std::declval<EleT>(), std::declval<EleTs>()...))>;
  return [](auto &&t, auto &&... ts) {
    return make_ewise_op_result<ele_t, ShapeT>(OpT(), wheels_forward(t),
                                               wheels_forward(ts)...);
  };
}

// tensor vs scalar
template <class OpT, class EleT1, class ShapeT1, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const ewise_base<EleT1, ShapeT1, T1> &,
                           const category::other<T2> &) {
  using ele_t =
      std::decay_t<decltype(OpT()(std::declval<EleT1>(), std::declval<T2>()))>;
  return [](auto &&t1, auto &&t2) {
    return make_ewise_op_result<ele_t, ShapeT1>(
        OpT()(const_symbol<0>(), wheels_forward(t2)), wheels_forward(t1));
  };
}

// scalar vs tensor
template <class OpT, class T1, class EleT2, class ShapeT2, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const category::other<T1> &,
                           const ewise_base<EleT2, ShapeT2, T2> &) {
  using ele_t =
      std::decay_t<decltype(OpT()(std::declval<T1>(), std::declval<EleT2>()))>;
  return [](auto &&t1, auto &&t2) {
    return make_ewise_op_result<ele_t, ShapeT2>(
        OpT()(wheels_forward(t1), const_symbol<0>()), wheels_forward(t2));
  };
}

// tensor vs const_expr
template <class OpT, class EleT1, class ShapeT1, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const ewise_base<EleT1, ShapeT1, T1> &,
                           const const_expr_base<T2> &) {
  return [](auto &&t1, auto &&t2) {
    return make_binary_op_expr(OpT(), as_const_coeff(wheels_forward(t1)),
                               wheels_forward(t2));
  };
}

// const_expr vs tensor
template <class OpT, class T1, class EleT2, class ShapeT2, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const const_expr_base<T1> &,
                           const ewise_base<EleT2, ShapeT2, T2> &) {
  return [](auto &&t1, auto &&t2) {
    return make_binary_op_expr(OpT(), wheels_forward(t1),
                               as_const_coeff(wheels_forward(t2)));
  };
}

namespace details {
// auto transform(ts)
template <class EleT, class ShapeT, class T, class FunT>
constexpr auto _transform(const tensor_base<EleT, ShapeT, T> &t, FunT &&fun) {
  using ele_t = std::decay_t<decltype(fun(std::declval<EleT>()))>;
  return make_ewise_op_result<ele_t, ShapeT>(std::forward<FunT>(fun),
                                             t.derived());
}
template <class EleT, class ShapeT, class T, class FunT>
constexpr auto _transform(tensor_base<EleT, ShapeT, T> &&t, FunT &&fun) {
  using ele_t = std::decay_t<decltype(fun(std::declval<EleT>()))>;
  return make_ewise_op_result<ele_t, ShapeT>(std::forward<FunT>(fun),
                                             std::move(t.derived()));
}

// cast
template <class TargetEleT, class EleT, class ShapeT, class T>
constexpr auto _static_ecast(const tensor_base<EleT, ShapeT, T> &t) {
  return _transform(t.derived(),
                    [](const auto &e) { return static_cast<TargetEleT>(e); })
}
template <class TargetEleT, class EleT, class ShapeT, class T>
constexpr auto _static_ecast(tensor_base<EleT, ShapeT, T> &&t) {
  return _transform(std::move(t.derived()),
                    [](const auto &e) { return static_cast<TargetEleT>(e); });
}
}

// transform
template <class EleT, class ShapeT, class T>
template <class FunT>
inline constexpr auto
ewise_base<EleT, ShapeT, T>::transform(FunT &&fun) const & {
  return details::_transform(derived(), std::forward<FunT>(fun));
}
template <class EleT, class ShapeT, class T>
template <class FunT>
inline auto ewise_base<EleT, ShapeT, T>::transform(FunT &&fun) & {
  return details::_transform(derived(), std::forward<FunT>(fun));
}
template <class EleT, class ShapeT, class T>
template <class FunT>
inline auto ewise_base<EleT, ShapeT, T>::transform(FunT &&fun) && {
  return details::_transform(std::move(derived()), std::forward<FunT>(fun));
}

// cast
template <class EleT, class ShapeT, class T>
template <class TargetEleT>
inline constexpr auto ewise_base<EleT, ShapeT, T>::cast() const & {
  return details::_static_ecast<TargetEleT>(derived());
}
template <class EleT, class ShapeT, class T>
template <class TargetEleT>
inline constexpr auto ewise_base<EleT, ShapeT, T>::cast() && {
  return details::_static_ecast<TargetEleT>(std::move(derived()));
}

// _as_tuple_seq
namespace details {
template <class FirstTT, class... TTs, size_t... Is, class FirstEleT,
          class... EleTs, class FirstShapeT, class... ShapeTs, class FirstT,
          class... Ts>
constexpr auto _as_tuple_seq(std::tuple<FirstTT, TTs...> &&ts,
                             const const_index<Is...> &,
                             const ewise_base<FirstEleT, FirstShapeT, FirstT> &,
                             const ewise_base<EleTs, ShapeTs, Ts> &...) {
  using result_ele_t = std::tuple<FirstEleT, EleTs...>;
  return make_ewise_op_result<result_ele_t, FirstShapeT>(
      [](auto &&... es) { return as_tuple(wheels_forward(es)...); },
      std::get<0>(ts), std::get<Is + 1>(ts)...);
}
}
}