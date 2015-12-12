#pragma once

#include "tensor.hpp"

namespace wheels {

// matrix_transpose
template <class ShapeT, class ET, class T>
class matrix_transpose
    : public tensor_base<ShapeT, ET, matrix_transpose<ShapeT, ET, T>> {
public:
  using value_type = ET;
  using shape_type = ShapeT;
  constexpr explicit matrix_transpose(T &&in) : _input(forward<T>(in)) {}
  constexpr const T &input() const { return _input; }
  T &input() { return _input; }
  constexpr auto shape() const {
    return make_shape(size_at(_input, const_index<1>()),
                      size_at(_input, const_index<0>()));
  }
  template <class SubT1, class SubT2>
  constexpr decltype(auto) at_subs(const SubT1 &s1, const SubT2 &s2) const {
    return element_at(_input, s2, s1);
  }

private:
  T _input;
};

// shape_of
template <class ShapeT, class ET, class T>
constexpr auto shape_of(const matrix_transpose<ShapeT, ET, T> &m) {
  return m.shape();
}
// element_at
template <class ShapeT, class ET, class T, class SubT1, class SubT2>
constexpr decltype(auto) element_at(const matrix_transpose<ShapeT, ET, T> &m,
                                    const SubT1 &s1, const SubT2 &s2) {
  return m.at_subs(s1, s2);
}
// for_each_element
template <class FunT, class ShapeT, class ET, class T>
void for_each_element(order_flag<unordered> o, FunT &&fun,
                      const matrix_transpose<ShapeT, ET, T> &m) {
  for_each_element(o, forward<FunT>(fun), m.input());
}

// for_each_element_if
template <class FunT, class ShapeT, class ET, class T>
bool for_each_element_with_short_circuit(
    order_flag<unordered> o, FunT &&fun,
    const matrix_transpose<ShapeT, ET, T> &m) {
  return for_each_element_with_short_circuit(o, forward<FunT>(fun), m.input());
}

// for_each_nonzero_element
template <class FunT, class ShapeT, class ET, class T>
void for_each_nonzero_element(order_flag<unordered> o, FunT &&fun,
                              const matrix_transpose<ShapeT, ET, T> &m) {
  for_each_nonzero_element(o, forward<FunT>(fun), m.input());
}
// reduce_elements
template <class ShapeT, class ET, class T, class E, class ReduceT>
E reduce_elements(const matrix_transpose<ShapeT, ET, T> &t, E initial,
                  ReduceT &&red) {
  return reduce_elements(t.input(), std::move(initial), forward<ReduceT>(red));
}
// norm_squared
template <class ShapeT, class ET, class T>
ET norm_squared(const matrix_transpose<ShapeT, ET, T> &t) {
  return norm_squared(t.input());
}
// bool all(s)
template <class ShapeT, class ET, class T>
constexpr bool all_of(const matrix_transpose<ShapeT, ET, T> &t) {
  return all_of(t.input());
}
// bool any(s)
template <class ShapeT, class ET, class T>
constexpr bool any_of(const matrix_transpose<ShapeT, ET, T> &t) {
  return any_of(t.input());
}

// transpose
template <class ST, class MT, class NT, class ET, class T>
constexpr auto
transpose(const tensor_base<tensor_shape<ST, MT, NT>, ET, T> &t) {
  return matrix_transpose<tensor_shape<ST, NT, MT>, ET, const T &>(t.derived());
}
template <class ST, class MT, class NT, class ET, class T>
auto transpose(tensor_base<tensor_shape<ST, MT, NT>, ET, T> &t) {
  return matrix_transpose<tensor_shape<ST, NT, MT>, ET, T &>(t.derived());
}
template <class ST, class MT, class NT, class ET, class T>
auto transpose(tensor_base<tensor_shape<ST, MT, NT>, ET, T> &&t) {
  return matrix_transpose<tensor_shape<ST, NT, MT>, ET, T>(
      std::move(t.derived()));
}
template <class ST, class MT, class NT, class ET, class T>
constexpr auto
transpose(const tensor_base<tensor_shape<ST, MT, NT>, ET, T> &&t) {
  return matrix_transpose<tensor_shape<ST, NT, MT>, ET, const T>(
      std::move(t.derived()));
}
// tranpose(matrix_transpose<...>)
template <class ShapeT, class ET, class T>
constexpr auto transpose(const matrix_transpose<ShapeT, ET, T> &t) {
  return t.input();
}
}
