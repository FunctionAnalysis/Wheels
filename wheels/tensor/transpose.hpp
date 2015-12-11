#pragma once

#include "tensor.hpp"

namespace wheels {

// transpose
template <class ShapeT, class ET, class T>
class matrix_transpose
    : public tensor_base<ShapeT, ET, matrix_transpose<ShapeT, ET, T>> {
public:
  using value_type = ET;
  using shape_type = ShapeT;
  constexpr explicit matrix_transpose(T &&in) : _input(forward<T>(in)) {}
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

template <class ST, class MT, class NT, class ET, class T>
constexpr auto
transpose(const tensor_base<tensor_shape<ST, MT, NT>, ET, T> &t) {
  return matrix_transpose<tensor_shape<ST, NT, MT>, ET, const T &>(t.derived());
}
template <class ST, class MT, class NT, class ET, class T>
constexpr auto transpose(tensor_base<tensor_shape<ST, MT, NT>, ET, T> &t) {
  return matrix_transpose<tensor_shape<ST, NT, MT>, ET, T &>(t.derived());
}
template <class ST, class MT, class NT, class ET, class T>
constexpr auto transpose(tensor_base<tensor_shape<ST, MT, NT>, ET, T> &&t) {
  return matrix_transpose<tensor_shape<ST, NT, MT>, ET, T>(
      std::move(t.derived()));
}
template <class ST, class MT, class NT, class ET, class T>
constexpr auto
transpose(const tensor_base<tensor_shape<ST, MT, NT>, ET, T> &&t) {
  return matrix_transpose<tensor_shape<ST, NT, MT>, ET, const T>(
      std::move(t.derived()));
}
}
