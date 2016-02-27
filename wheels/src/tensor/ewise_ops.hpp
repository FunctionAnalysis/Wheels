#pragma once

#include "tensor.hpp"

namespace wheels {

// ewise ops
template <class EleT, class ShapeT, class OpT, class InputT, class... InputTs>
class ewise_op_result
    : public tensor_op_result_base<
          EleT, ShapeT, OpT,
          ewise_op_result<EleT, ShapeT, OpT, InputT, InputTs...>> {
public:
  using shape_type = ShapeT;
  using value_type = EleT;
  constexpr explicit ewise_op_result(OpT o, InputT &&in, InputTs &&... ins)
      : op(o), inputs(forward<InputT>(in), forward<InputTs>(ins)...) {}

public:
  OpT op;
  std::tuple<InputT, InputTs...> inputs;
};

// make_ewise_op_result
template <class ET, class ShapeT, class OpT, class InputT, class... InputTs>
constexpr ewise_op_result<ET, ShapeT, OpT, InputT, InputTs...>
make_ewise_op_result(OpT op, InputT &&input, InputTs &&... inputs) {
  return ewise_op_result<ET, ShapeT, OpT, InputT, InputTs...>(
      op, forward<InputT>(input), forward<InputTs>(inputs)...);
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



// other ops are overloaded as ewise operation results defaultly
// all tensors
template <class OpT, class EleT, class ShapeT, class T, class... ShapeTs,
          class... EleTs, class... Ts>
struct overloaded<OpT, category_tensor<EleT, ShapeT, T>,
                  category_tensor<EleTs, ShapeTs, Ts>...> {
  template <class TT, class... TTs>
  constexpr decltype(auto) operator()(TT &&t, TTs &&... ts) const {
    assert((std::is_same<OpT, binary_op_eq>::value ||
            std::is_same<OpT, binary_op_neq>::value ||
            all_same(shape_of(t), shape_of(ts)...)));
    using ele_t = std::decay_t<decltype(
        OpT()(std::declval<EleT>(), std::declval<EleTs>()...))>;
    return make_ewise_op_result<ele_t, ShapeT>(OpT(), forward<TT>(t),
                                               forward<TTs>(ts)...);
  }
};

template <class EleT, class ShapeT, class T, class ShapeT2, class EleT2,
          class T2>
struct overloaded<binary_op_mul, category_tensor<EleT, ShapeT, T>,
                  category_tensor<EleT2, ShapeT2, T2>> {
  template <class TT, class TT2>
  constexpr int operator()(TT &&t, TT2 &&t2) const {
    static_assert(always<bool, false, TT, TT2>::value,
                  "use ewise_mul(t1, t2) if you want to compute element-wise "
                  "product of two tensors");
  }
};

template <class EleT, class ShapeT, class T, class ShapeT2, class EleT2,
          class T2>
struct overloaded<ewised<binary_op_mul>, category_tensor<EleT, ShapeT, T>,
                  category_tensor<EleT2, ShapeT2, T2>> {
  template <class TT, class TT2>
  constexpr decltype(auto) operator()(TT &&t, TT2 &&t2) const {
    using ele_t = decltype(std::declval<EleT>() * std::declval<EleT2>());
    return make_ewise_op_result<ele_t, ShapeT>(binary_op_mul(), forward<TT>(t),
                                               forward<TT2>(t2));
  }
};

// tensor vs scalar
template <class OpT, class EleT, class ShapeT, class T>
struct overloaded<OpT, category_tensor<EleT, ShapeT, T>, void> {
  template <class T1, class T2>
  constexpr decltype(auto) operator()(T1 &&t1, T2 &&t2) const {
    using ele_t =
        std::decay_t<decltype(OpT()(std::declval<EleT>(), std::declval<T2>()))>;
    return make_ewise_op_result<ele_t, ShapeT>(
        OpT()(const_symbol<0>(), forward<T2>(t2)), forward<T1>(t1));
  }
};
// scalar vs tensor
template <class OpT, class EleT, class ShapeT, class T>
struct overloaded<OpT, void, category_tensor<EleT, ShapeT, T>> {
  template <class T1, class T2>
  constexpr decltype(auto) operator()(T1 &&t1, T2 &&t2) const {
    using ele_t =
        std::decay_t<decltype(OpT()(std::declval<T1>(), std::declval<EleT>()))>;
    return make_ewise_op_result<ele_t, ShapeT>(
        OpT()(forward<T1>(t1), const_symbol<0>()), forward<T2>(t2));
  }
};
// tensor vs const_expr
template <class OpT, class EleT, class ShapeT, class T>
struct overloaded<OpT, category_tensor<EleT, ShapeT, T>, category_const_expr> {
  template <class T1, class T2>
  constexpr decltype(auto) operator()(T1 &&t1, T2 &&t2) const {
    return const_binary_op<OpT, const_coeff<std::decay_t<T1>>, T2>(
        OpT(), as_const_coeff(forward<T1>(t1)), forward<T2>(t2));
  }
};
// const_expr vs tensor
template <class OpT, class EleT, class ShapeT, class T>
struct overloaded<OpT, category_const_expr, category_tensor<EleT, ShapeT, T>> {
  template <class T1, class T2>
  constexpr decltype(auto) operator()(T1 &&t1, T2 &&t2) const {
    return const_binary_op<OpT, T1, const_coeff<std::decay_t<T2>>>(
        OpT(), forward<T1>(t1), as_const_coeff(forward<T2>(t2)));
  }
};

// auto transform(ts)
template <class EleT, class ShapeT, class T, class FunT>
constexpr auto transform(const tensor_base<EleT, ShapeT, T> &t, FunT &&fun) {
  using ele_t = std::decay_t<decltype(fun(std::declval<EleT>()))>;
  return make_ewise_op_result<ele_t, ShapeT>(std::forward<FunT>(fun),
                                             t.derived());
}
template <class EleT, class ShapeT, class T, class FunT>
constexpr auto transform(tensor_base<EleT, ShapeT, T> &&t, FunT &&fun) {
  using ele_t = std::decay_t<decltype(fun(std::declval<EleT>()))>;
  return make_ewise_op_result<ele_t, ShapeT>(std::forward<FunT>(fun),
                                             std::move(t.derived()));
}

// cast
template <class TargetEleT, class EleT, class ShapeT, class T>
constexpr auto static_ecast(const tensor_base<EleT, ShapeT, T> &t) {
  return transform(t.derived(),
                   [](const auto &e) { return static_cast<TargetEleT>(e); })
}
template <class TargetEleT, class EleT, class ShapeT, class T>
constexpr auto static_ecast(tensor_base<EleT, ShapeT, T> &&t) {
  return transform(std::move(t.derived()),
                   [](const auto &e) { return static_cast<TargetEleT>(e); });
}

// auto normalize(ts)
template <class T> constexpr auto normalize(T &&t) {
  return forward<T>(t) / norm(t);
}
}