#pragma once

#include "tensor.hpp"

namespace wheels {

namespace details {
template <class ShapeT1, class ShapeT2, size_t Axis, size_t... Is>
constexpr auto _make_cat_shape_seq(const ShapeT1 &s1, const ShapeT2 &s2,
                                   const const_index<Axis> &axis,
                                   const const_ints<size_t, Is...> &) {
  static_assert(ShapeT1::rank == ShapeT2::rank, "shape ranks mismatch");
  static_assert(Axis < ShapeT1::rank, "Axis value overflow");
  assert(::wheels::all(s1.at(const_index<Is>()) == s2.at(const_index<Is>()) ||
                       Is == Axis...) &&
         "shape sizes should equal except at Axis when performing tensor cat");
  return make_shape(
      ::wheels::conditional(axis == const_index<Is>(),
                            s1.at(const_index<Is>()) + s2.at(const_index<Is>()),
                            s1.at(const_index<Is>()))...);
}
}

// cat_result
template <class ShapeT, class ET, size_t Axis, class T1, class T2>
class cat_result
    : public tensor_op_result_base<ShapeT, ET, void,
                                   cat_result<ShapeT, ET, Axis, T1, T2>> {
public:
  using value_type = ET;
  using shape_type = ShapeT;

  cat_result(T1 &&in1, T2 &&in2)
      : _input1(forward<T1>(in1)), _input2(forward<T2>(in2)),
        _shape(details::_make_cat_shape_seq(in1.shape(), in2.shape(),
                                            const_index<Axis>(),
                                            make_rank_sequence(in1.shape()))) {}
  constexpr decltype(auto) input1() const { return _input1; }
  constexpr decltype(auto) input2() const { return _input2; }
  constexpr const ShapeT &shape() const { return _shape; }

private:
  T1 _input1;
  T2 _input2;
  ShapeT _shape;
};

// cat_at
namespace details {
template <size_t Axis, class ShapeT1, class ET1, class T1, class ShapeT2,
          class ET2, class T2, class TT1, class TT2>
constexpr auto _cat_tensor_at(const const_index<Axis> &axis,
                              const tensor_base<ShapeT1, ET1, T1> &,
                              const tensor_base<ShapeT2, ET2, T2> &, TT1 &&in1,
                              TT2 &in2) {
  using shape_t = decltype(details::_make_cat_shape_seq(
      in1.shape(), in2.shape(), const_index<Axis>(),
      make_rank_sequence(in1.shape())));
  using ele_t = std::common_type_t<ET1, ET2>;
  return cat_result<shape_t, ele_t, Axis, TT1, TT2>(forward<TT1>(in1),
                                                    forward<TT2>(in2));
}
}
template <size_t Axis, class T1, class T2>
constexpr auto cat_at(const const_index<Axis> &axis, T1 &&in1, T2 &&in2)
    -> decltype(details::_cat_tensor_at(axis, in1, in2, forward<T1>(in1),
                                        forward<T2>(in2))) {
  return details::_cat_tensor_at(axis, in1, in2, forward<T1>(in1),
                                 forward<T2>(in2));
}

// cat2 (cat_at 0)
template <class T1, class T2>
constexpr auto cat2(T1 &&in1, T2 &&in2)
    -> decltype(cat_at(const_index<0>(), forward<T1>(in1), forward<T2>(in2))) {
  return cat_at(const_index<0>(), forward<T1>(in1), forward<T2>(in2));
}

// shape_of
template <class ShapeT, class ET, size_t Axis, class T1, class T2>
constexpr auto shape_of(const cat_result<ShapeT, ET, Axis, T1, T2> &m) {
  return m.shape();
}

// element_at
namespace details {
template <class ShapeT, class ET, size_t Axis, class T1, class T2,
          class SubsTupleT, size_t... Is>
constexpr ET
_element_at_cat_result_seq(const cat_result<ShapeT, ET, Axis, T1, T2> &m,
                           SubsTupleT &subs,
                           const const_ints<size_t, Is...> &) {
  return (ET)conditional(
      std::get<Axis>(subs) < m.input1().shape().at(const_index<Axis>()),
      element_at(m.input1(), std::get<Is>(subs)...),
      element_at(m.input2(),
                 conditional(const_bool<Axis == Is>(),
                             std::get<Axis>(subs) -
                                 m.input1().shape().at(const_index<Axis>()),
                             std::get<Is>(subs))...));
}
}
template <class ShapeT, class ET, size_t Axis, class T1, class T2,
          class... SubTs>
constexpr ET element_at(const cat_result<ShapeT, ET, Axis, T1, T2> &m,
                        const SubTs &... subs) {
  return details::_element_at_cat_result_seq(
      m, std::forward_as_tuple(subs...),
      make_rank_sequence(m.input2().shape()));
}
}
