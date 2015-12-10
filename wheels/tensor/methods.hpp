#pragma once

#include "tensor.hpp"

namespace wheels {

// distance
template <class ShapeT1, class ET1, class T1, class ShapeT2, class ET2,
          class T2>
constexpr auto distance(const tensor_base<ShapeT1, ET1, T1> &t1,
                        const tensor_base<ShapeT2, ET2, T2> &t2) {
  return norm(t1.derived() - t2.derived());
}

// dot(ts1, ts2);
template <class ShapeT1, class ET1, class T1, class ShapeT2, class ET2,
          class T2>
auto dot(const tensor_base<ShapeT1, ET1, T1> &t1,
         const tensor_base<ShapeT2, ET2, T2> &t2) {
  using result_t = std::common_type_t<ET1, ET2>;
  assert(shape_of(t1.derived()) == shape_of(t2.derived()));
  result_t result = 0.0;
  for_each_element([&result](auto &&e1, auto &&e2) { result += e1 * e2; },
                   t1.derived(), t2.derived());
  return result;
}

// auto cross(ts1, ts2);
template <class ST1, class NT1, class E1, class T1, class ST2, class NT2,
          class E2, class T2>
constexpr auto cross(const tensor_base<tensor_shape<ST1, NT1>, E1, T1> &a,
                     const tensor_base<tensor_shape<ST2, NT2>, E2, T2> &b) {
  using result_t = std::common_type_t<E1, E2>;
  return vec_<result_t, 3>(a.y() * b.z() - a.z() * b.y(),
                           a.z() * b.x() - a.x() * b.z(),
                           a.x() * b.y() - a.y() * b.x());
}

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
  template <wheels_enable_if((std::is_same<ET, bool>::value))>
  constexpr operator bool() const {
    return !!_val;
  }

private:
  ShapeT _shape;
  ET _val;
};

template <class ShapeT, class ET>
constexpr auto shape_of(const constant_result<ShapeT, ET> &t) {
  return t.shape();
}

template <class ShapeT, class ET, class... SubTs>
constexpr const ET &element_at(const constant_result<ShapeT, ET> &t,
                               const SubTs &... subs) {
  return t.value();
}

template <class ShapeT, class ET, class IndexT>
constexpr const ET &element_at_index(const constant_result<ShapeT, ET> &t,
                                     const IndexT &ind) {
  return t.value();
}

template <class FunT, class ShapeT, class ET, class... Ts>
void for_each_nonzero_element(FunT &&fun, const constant_result<ShapeT, ET> &t,
                              Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  if (t.value()) {
    for_each_element(forward<FunT>(fun), t, forward<Ts>(ts)...);
  }
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

// fast implementation of ewise ops
// all constants
template <class OpT, class ShapeT, class EleT, class... ShapeTs, class... EleTs>
struct overloaded<
    OpT, category_tensor<ShapeT, EleT, constant_result<ShapeT, EleT>>,
    category_tensor<ShapeTs, EleTs, constant_result<ShapeTs, EleTs>>...> {
  template <class TT, class... TTs>
  constexpr decltype(auto) operator()(TT &&t, TTs &&... ts) const {
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

// tensor vs scalar
template <class OpT, class ShapeT, class EleT>
struct overloaded<
    OpT, category_tensor<ShapeT, EleT, constant_result<ShapeT, EleT>>, void> {
  template <class T1, class T2>
  constexpr decltype(auto) operator()(T1 &&t1, T2 &&t2) const {
    return constants(t1.shape(), OpT()(t1.value(), forward<T2>(t2)));
  }
};
// scalar vs tensor
template <class OpT, class ShapeT, class EleT>
struct overloaded<
    OpT, void, category_tensor<ShapeT, EleT, constant_result<ShapeT, EleT>>> {
  template <class T1, class T2>
  constexpr decltype(auto) operator()(T1 &&t1, T2 &&t2) const {
    return constants(t2.shape(), OpT()(forward<T1>(t1), t2.value()));
  }
};


// eye


// iota

// meshgrid




}