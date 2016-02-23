#pragma once

#include "base.hpp"
#include "constants.hpp"

namespace wheels {

// diagonal_result
template <class ET, class ShapeT, class T>
class diagonal_result
    : public tensor_base<ET, ShapeT, diagonal_result<ET, ShapeT, T>> {
public:
  constexpr explicit diagonal_result(const ShapeT &s, T &&in)
      : _shape(s), _input(forward<T>(in)) {}
  constexpr const ShapeT &shape() const { return _shape; }
  constexpr const T &input() const & { return _input; }
  T &input() & { return _input; }
  T &&input() && { return _input; }

private:
  ShapeT _shape;
  T _input;
};

// shape_of
template <class ET, class ShapeT, class T>
constexpr decltype(auto) shape_of(const diagonal_result<ET, ShapeT, T> &t) {
  return t.shape();
}

// element_at
template <class ET, class ShapeT, class T, class SubT, class... SubTs>
constexpr decltype(auto) element_at(const diagonal_result<ET, ShapeT, T> &t,
                                    const SubT &sub, const SubTs &... subs) {
  return conditional(all_same(sub, subs...), element_at_index(t.input(), sub),
                     types<ET>::zero());
}

// TODO
// optimize for for_each(nonzero), sum, norm_squared ...

namespace details {
template <class ET, class ShapeT, class T, class TT, class NewShapeT>
constexpr auto _diag(const tensor_base<ET, ShapeT, T> &, TT &&t,
                     const NewShapeT &nshape) {
  assert(t.numel() == min_shape_size(nshape));
  return diagonal_result<ET, NewShapeT, TT>(nshape, forward<TT>(t));
}
}

// diag
template <class T, class ST, class... SizeTs>
constexpr auto diag(T &&t, const tensor_shape<ST, SizeTs...> &ns)
    -> decltype(details::_diag(t, forward<T>(t), ns)) {
  return details::_diag(t, forward<T>(t), ns);
}
template <class T>
constexpr auto diag(T &&t)
    -> decltype(details::_diag(t, forward<T>(t),
                               make_shape(t.numel(), t.numel()))) {
  return details::_diag(t, forward<T>(t), make_shape(t.numel(), t.numel()));
}

// eye
template <class ET = double, class ST, class... SizeTs>
constexpr auto eye(const tensor_shape<ST, SizeTs...> &s) {
  return diag(ones<ET>(make_shape(min_shape_size(s))), s);
}
template <class ET = double, class MT, class NT>
constexpr auto eye(const MT &m, const NT &n) {
  return diag(ones<ET>(make_shape(min(m, n))), make_shape(m, n));
}
template <class ET = double, class NT,
          class = std::enable_if_t<!is_tensor_shape<NT>::value>>
constexpr auto eye(const NT &n) {
  return diag(ones<ET>(make_shape(n)), make_shape(n, n));
}
}
