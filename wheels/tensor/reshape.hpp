#pragma once

#include "base.hpp"

namespace wheels {

// reshape_view
template <class ET, class ShapeT, class T>
class reshape_view
    : public tensor_base<ET, ShapeT, reshape_view<ET, ShapeT, T>> {
public:
  using value_type = ET;
  using shape_type = ShapeT;
  constexpr reshape_view(const ShapeT &s, T &&in)
      : _shape(s), _input(forward<T>(in)) {
    assert(s.magnitude() <= _input.numel());
  }

  const ShapeT &shape() const { return _shape; }
  const T &input() const & { return _input; }
  T &input() & { return _input; }
  T &&input() && { return _input; }

private:
  ShapeT _shape;
  T _input;
};

// shape_of
template <class ET, class ShapeT, class T>
constexpr const ShapeT &shape_of(const reshape_view<ET, ShapeT, T> &t) {
  return t.shape();
}

// element_at
template <class ET, class ShapeT, class T, class... SubTs>
constexpr decltype(auto) element_at(const reshape_view<ET, ShapeT, T> &t,
                                    const SubTs &... subs) {
  return element_at_index(t.input(), sub2ind(t.shape(), subs...));
}
template <class ET, class ShapeT, class T, class... SubTs>
decltype(auto) element_at(reshape_view<ET, ShapeT, T> &t,
                          const SubTs &... subs) {
  return element_at_index(t.input(), sub2ind(t.shape(), subs...));
}

// element_at_index
template <class ET, class ShapeT, class T, class IndexT>
constexpr decltype(auto) element_at_index(const reshape_view<ET, ShapeT, T> &t,
                                          const IndexT &ind) {
  return element_at_index(t.input(), ind);
}
template <class ET, class ShapeT, class T, class IndexT>
decltype(auto) element_at_index(reshape_view<ET, ShapeT, T> &t,
                                const IndexT &ind) {
  return element_at_index(t.input(), ind);
}


// reshape
namespace details {
template <class ET, class OldShapeT, class T, class TT, class ShapeT>
constexpr auto _reshape(const tensor_base<ET, OldShapeT, T> &, TT &&t,
                        const ShapeT &s) {
  return reshape_view<ET, ShapeT, TT>(s, forward<TT>(t));
}
template <class ET, class OldShapeT, class T, class TT, class ShapeT>
constexpr auto _reshape(const reshape_view<ET, OldShapeT, T> &, TT &&t,
                        const ShapeT &s) {
  return reshape_view<ET, ShapeT, T>(
      s, forward<TT>(t).input()); // todo: needs investigation on how to
                                  // perfectly forwarding class members
}
}

template <class T, class ST, class... SizeTs>
constexpr auto reshape(T &&t, const tensor_shape<ST, SizeTs...> &s)
    -> decltype(details::_reshape(t, forward<T>(t), s)) {
  return details::_reshape(t, forward<T>(t), s);
}
}