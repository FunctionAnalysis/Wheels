#pragma once

#include "../core/const_ints.hpp"

#include "base.hpp"

#include "constants_fwd.hpp"

namespace wheels {

// constant_result
template <class ET, class ShapeT, class OpT>
class constant_result
    : public tensor_op_result_base<ET, ShapeT, OpT,
                                   constant_result<ET, ShapeT, OpT>> {
public:
  using shape_type = ShapeT;
  using value_type = ET;
  template <class EE>
  constexpr explicit constant_result(const ShapeT &s, EE &&v)
      : _shape(s), _val(std::forward<EE>(v)) {}
  const ShapeT &shape() const { return _shape; }
  const ET &value() const { return _val; }
  ET &value() { return _val; }

private:
  ShapeT _shape;
  ET _val;
};

// shape_of
template <class ET, class ShapeT, class OpT>
constexpr const ShapeT &shape_of(const constant_result<ET, ShapeT, OpT> &t) {
  return t.shape();
}

// element_at
template <class ET, class ShapeT, class OpT, class... SubTs>
constexpr const ET &element_at(const constant_result<ET, ShapeT, OpT> &t,
                               const SubTs &... subs) {
  return t.value();
}

// element_at_index
template <class ET, class ShapeT, class OpT, class IndexT>
constexpr const ET &element_at_index(const constant_result<ET, ShapeT, OpT> &t,
                                     const IndexT &ind) {
  return t.value();
}

// index_ascending, unordered
template <behavior_flag_enum O, class FunT, class ET, class ShapeT, class OpT>
bool for_each_element(behavior_flag<O>, FunT &&fun,
                      const constant_result<ET, ShapeT, OpT> &t) {
  for (size_t i = 0; i < numel(t); i++) {
    fun(t.value());
  }
  return true;
}

// break_on_false
template <class FunT, class ET, class ShapeT, class OpT>
bool for_each_element(behavior_flag<break_on_false>, FunT &&fun,
                      const constant_result<ET, ShapeT, OpT> &t) {
  for (size_t i = 0; i < numel(t); i++) {
    if (!fun(t.value())) {
      return false;
    }
  }
  return true;
}

// nonzero_only
template <class FunT, class ET, class ShapeT, class OpT, class... Ts>
bool for_each_element(behavior_flag<nonzero_only> o, FunT &&fun,
                      const constant_result<ET, ShapeT, OpT> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  if (!is_zero(t.value())) {
    return for_each_element(behavior_flag<unordered>(), std::forward<FunT>(fun), t,
                            std::forward<Ts>(ts)...);
  }
  return false;
}

// size_t nonzero_elements_count(t)
template <class ET, class ShapeT, class OpT>
size_t nonzero_elements_count(const constant_result<ET, ShapeT, OpT> &t) {
  return is_zero(t.value()) ? 0 : numel(t);
}

// reduce_elements
template <class ET, class ShapeT, class OpT, class E, class ReduceT>
E reduce_elements(const constant_result<ET, ShapeT, OpT> &t, E initial,
                  ReduceT &&red) {
  for (size_t i = 0; i < numel(t); i++) {
    initial = red(initial, t.value());
  }
  return initial;
}

// sum_of
template <class ET, class ShapeT, class OpT>
std::enable_if_t<std::is_scalar<ET>::value, ET>
sum_of(const constant_result<ET, ShapeT, OpT> &t) {
  return t.value() * numel(t);
}

// norm_squared
template <class ET, class ShapeT, class OpT>
std::enable_if_t<std::is_scalar<ET>::value, ET>
norm_squared(const constant_result<ET, ShapeT, OpT> &t) {
  return t.value() * t.value() * numel(t);
}

// all_of
template <class ET, class ShapeT, class OpT>
constexpr bool all_of(const constant_result<ET, ShapeT, OpT> &t) {
  return !!t.value();
}

// any_of
template <class ET, class ShapeT, class OpT>
constexpr bool any_of(const constant_result<ET, ShapeT, OpT> &t) {
  return !!t.value();
}

// constants
template <class ET, class ST, class... SizeTs>
constexpr auto constants(const tensor_shape<ST, SizeTs...> &shape, ET &&v) {
  return constant_result<std::decay_t<ET>, tensor_shape<ST, SizeTs...>, void>(
      shape, std::forward<ET>(v));
}

// zeros
template <class ET, class ST, class... SizeTs>
constexpr auto zeros(const tensor_shape<ST, SizeTs...> &shape) {
  return constants<ET>(shape, 0);
}
template <class ET, class... SizeTs>
constexpr auto zeros(const SizeTs &... sizes) {
  return constants<ET>(make_shape(sizes...), 0);
}

// ones
template <class ET, class ST, class... SizeTs>
constexpr auto ones(const tensor_shape<ST, SizeTs...> &shape) {
  return constants<ET>(shape, 1);
}
template <class ET, class... SizeTs>
constexpr auto ones(const SizeTs &... sizes) {
  return constants<ET>(make_shape(sizes...), 1);
}

namespace details {
template <class ET, class ST, class... SizeTs, class OpT>
constexpr auto _constants(const tensor_shape<ST, SizeTs...> &shape, ET &&v,
                          OpT &&) {
  return constant_result<std::decay_t<ET>, tensor_shape<ST, SizeTs...>,
                         std::decay_t<OpT>>(shape, std::forward<ET>(v));
}
}

//// ewise ops
//// all constants
// template <class OpT, class EleT, class ShapeT, class COpT, class... ShapeTs,
//          class... OpTs, class... EleTs>
// struct overloaded<
//    OpT, category_tensor<EleT, ShapeT, constant_result<EleT, ShapeT, COpT>>,
//    category_tensor<EleTs, ShapeTs, constant_result<EleTs, ShapeTs, OpTs>>...>
//    {
//  template <class TT, class... TTs>
//  constexpr auto operator()(TT &&t, TTs &&... ts) const {
//    assert(all_same(shape_of(t), shape_of(ts)...));
//    return details::_constants(t.shape(), OpT()(t.value(), ts.value()...),
//                               OpT());
//  }
//};
//
// template <class EleT, class ShapeT, class OpT, class ShapeT2, class EleT2,
//          class OpT2>
// struct overloaded<
//    binary_op_mul,
//    category_tensor<EleT, ShapeT, constant_result<EleT, ShapeT, OpT>>,
//    category_tensor<EleT2, ShapeT2, constant_result<EleT2, ShapeT2, OpT2>>> {
//  template <class TT, class TT2>
//  constexpr int operator()(TT &&t, TT2 &&t2) const {
//    static_assert(always<bool, false, TT, TT2>::value,
//                  "use ewise_mul(t1, t2) if you want to compute element-wise "
//                  "product of two tensors");
//  }
//};
// template <class EleT, class ShapeT, class OpT, class ShapeT2, class EleT2,
//          class OpT2>
// struct overloaded<
//    ewised<binary_op_mul>,
//    category_tensor<EleT, ShapeT, constant_result<EleT, ShapeT, OpT>>,
//    category_tensor<EleT2, ShapeT2, constant_result<EleT2, ShapeT2, OpT2>>> {
//  template <class TT, class TT2>
//  constexpr auto operator()(TT &&t, TT2 &&t2) const {
//    assert(all_same(shape_of(t), shape_of(t2)));
//    return details::_constants(t.shape(), t.value() * t2.value(),
//                               ewised<binary_op_mul>());
//  }
//};
//
//// tensor vs scalar
// template <class OpT, class EleT, class ShapeT, class COpT>
// struct overloaded<
//    OpT, category_tensor<EleT, ShapeT, constant_result<EleT, ShapeT, COpT>>,
//    void> {
//  template <class T1, class T2>
//  constexpr auto operator()(T1 &&t1, T2 &&t2) const {
//    return details::_constants(t1.shape(), OpT()(t1.value(), std::forward<T2>(t2)),
//                               OpT());
//  }
//};
//// scalar vs tensor
// template <class OpT, class EleT, class ShapeT, class COpT>
// struct overloaded<
//    OpT, void,
//    category_tensor<EleT, ShapeT, constant_result<EleT, ShapeT, COpT>>> {
//  template <class T1, class T2>
//  constexpr auto operator()(T1 &&t1, T2 &&t2) const {
//    return details::_constants(t2.shape(), OpT()(std::forward<T1>(t1), t2.value()),
//                               OpT());
//  }
//};
}