#pragma once

#include "tensor.hpp"

namespace wheels {

// constant_result
template <class ShapeT, class ET>
class constant_result
    : public tensor_base<ShapeT, ET, constant_result<ShapeT, ET>> {
public:
  using shape_type = ShapeT;
  using value_type = ET;
  template <class EE>
  constexpr explicit constant_result(const ShapeT &s, EE &&v)
      : _shape(s), _val(forward<EE>(v)) {}
  const ShapeT &shape() const { return _shape; }
  const ET &value() const { return _val; }

private:
  ShapeT _shape;
  ET _val;
};

// shape_of
template <class ShapeT, class ET>
constexpr auto shape_of(const constant_result<ShapeT, ET> &t) {
  return t.shape();
}

// element_at
template <class ShapeT, class ET, class... SubTs>
constexpr const ET &element_at(const constant_result<ShapeT, ET> &t,
                               const SubTs &... subs) {
  return t.value();
}

// element_at_index
template <class ShapeT, class ET, class IndexT>
constexpr const ET &element_at_index(const constant_result<ShapeT, ET> &t,
                                     const IndexT &ind) {
  return t.value();
}

// for_each_element
template <order_flag_enum O, class FunT, class ShapeT, class ET>
void for_each_element(order_flag<O>, FunT &&fun, const constant_result<ShapeT, ET> &t) {
  for (size_t i = 0; i < numel(t); i++) {
    fun(t.value());
  }
}

// for_each_element_with_short_circuit
template <order_flag_enum O, class FunT, class ShapeT, class ET>
bool for_each_element_with_short_circuit(order_flag<O>, FunT &&fun, const constant_result<ShapeT, ET> &t) {
  for (size_t i = 0; i < numel(t); i++) {
    if (!fun(t.value())) {
      return false;
    }
  }
  return true;
}

// for_each_nonzero_element
template <order_flag_enum O, class FunT, class ShapeT, class ET, class... Ts>
void for_each_nonzero_element(order_flag<O> o, FunT &&fun, const constant_result<ShapeT, ET> &t,
                              Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  if (t.value()) {
    for_each_element(o, forward<FunT>(fun), t, forward<Ts>(ts)...);
  }
}

// reduce_elements
template <class ShapeT, class ET, class E, class ReduceT>
E reduce_elements(const constant_result<ShapeT, ET> &t, E initial,
                  ReduceT &&red) {
  for (size_t i = 0; i < numel(t); i++) {
    initial = red(initial, t.value());
  }
  return initial;
}

// norm_squared
template <class ShapeT, class ET>
std::enable_if_t<std::is_arithmetic<ET>::value, ET>
norm_squared(const constant_result<ShapeT, ET> &t) {
  return t.value() * t.value() * numel(t);
}

// all_of
template <class ShapeT, class ET>
constexpr bool all_of(const constant_result<ShapeT, ET> &t) {
  return !!t.value();
}

// any_of
template <class ShapeT, class ET>
constexpr bool any_of(const constant_result<ShapeT, ET> &t) {
  return !!t.value();
}

// constants
template <class ET, class ST, class... SizeTs>
constexpr auto constants(const tensor_shape<ST, SizeTs...> &shape, ET &&v) {
  return constant_result<tensor_shape<ST, SizeTs...>, std::decay_t<ET>>(
      shape, forward<ET>(v));
}

// zeros
template <class ET = double, class ST, class... SizeTs>
constexpr auto zeros(const tensor_shape<ST, SizeTs...> &shape) {
  return constants<ET>(shape, 0);
}
template <class ET = double, class... SizeTs>
constexpr auto zeros(const SizeTs &... sizes) {
  return constants<ET>(make_shape(sizes...), 0);
}

// ones
template <class ET = double, class ST, class... SizeTs>
constexpr auto ones(const tensor_shape<ST, SizeTs...> &shape) {
  return constants<ET>(shape, 1);
}
template <class ET = double, class... SizeTs>
constexpr auto ones(const SizeTs &... sizes) {
  return constants<ET>(make_shape(sizes...), 1);
}

// ewise ops
// all constants
template <class OpT, class ShapeT, class EleT, class... ShapeTs, class... EleTs>
struct overloaded<
    OpT, category_tensor<ShapeT, EleT, constant_result<ShapeT, EleT>>,
    category_tensor<ShapeTs, EleTs, constant_result<ShapeTs, EleTs>>...> {
  template <class TT, class... TTs>
  constexpr auto operator()(TT &&t, TTs &&... ts) const {
    assert(all_same(shape_of(t), shape_of(ts)...));
    return constants(t.shape(), OpT()(t.value(), ts.value()...));
  }
};
template <class ShapeT, class EleT, class ShapeT2, class EleT2>
struct overloaded<
    binary_op_mul, category_tensor<ShapeT, EleT, constant_result<ShapeT, EleT>>,
    category_tensor<ShapeT2, EleT2, constant_result<ShapeT2, EleT2>>> {
  template <class TT, class TT2>
  constexpr int operator()(TT &&t, TT2 &&t2) const {
    static_assert(always<bool, false, TT, TT2>::value,
                  "use ewise_mul(t1, t2) if you want to compute element-wise "
                  "product of two tensors");
  }
};
template <class ShapeT, class EleT, class ShapeT2, class EleT2>
struct overloaded<
    ewised<binary_op_mul>,
    category_tensor<ShapeT, EleT, constant_result<ShapeT, EleT>>,
    category_tensor<ShapeT2, EleT2, constant_result<ShapeT2, EleT2>>> {
  template <class TT, class TT2>
  constexpr auto operator()(TT &&t, TT2 &&t2) const {
    assert(all_same(shape_of(t), shape_of(t2)));
    return constants(t.shape(), t.value() * t2.value());
  }
};

// tensor vs scalar
template <class OpT, class ShapeT, class EleT>
struct overloaded<
    OpT, category_tensor<ShapeT, EleT, constant_result<ShapeT, EleT>>, void> {
  template <class T1, class T2>
  constexpr auto operator()(T1 &&t1, T2 &&t2) const {
    return constants(t1.shape(), OpT()(t1.value(), forward<T2>(t2)));
  }
};
// scalar vs tensor
template <class OpT, class ShapeT, class EleT>
struct overloaded<
    OpT, void, category_tensor<ShapeT, EleT, constant_result<ShapeT, EleT>>> {
  template <class T1, class T2>
  constexpr auto operator()(T1 &&t1, T2 &&t2) const {
    return constants(t2.shape(), OpT()(forward<T1>(t1), t2.value()));
  }
};
}