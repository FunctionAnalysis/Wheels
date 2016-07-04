#pragma once

#include "tensor_base.hpp"
#include "tensor_view_base.hpp"

#include "reshape_fwd.hpp"

namespace wheels {

// reshape_view
template <class ET, class ShapeT, class T>
class reshape_view
    : public tensor_view_base<ET, ShapeT, reshape_view<ET, ShapeT, T>, false> {
  using _base_t =
      tensor_view_base<ET, ShapeT, reshape_view<ET, ShapeT, T>, false>;

public:
  using value_type = ET;
  using shape_type = ShapeT;
  constexpr reshape_view(const ShapeT &s, T &&in)
      : _shape(s), _input(std::forward<T>(in)) {
    assert(s.magnitude() == _input.numel());
  }

  using _base_t::operator=;
  using _base_t::operator+=;
  using _base_t::operator-=;
  using _base_t::operator*=;
  using _base_t::operator/=;

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
  assert(subscripts_are_valid(t.shape(), subs...));
  return element_at_index(t.input(), sub2ind(t.shape(), subs...));
}
template <class ET, class ShapeT, class T, class... SubTs>
decltype(auto) element_at(reshape_view<ET, ShapeT, T> &t,
                          const SubTs &... subs) {
  assert(subscripts_are_valid(t.shape(), subs...));
  return element_at_index(t.input(), sub2ind(t.shape(), subs...));
}

// element_at_index
template <class ET, class ShapeT, class T, class IndexT>
constexpr decltype(auto) element_at_index(const reshape_view<ET, ShapeT, T> &t,
                                          const IndexT &ind) {
  assert(is_between(ind, 0, (IndexT)t.numel()));
  return element_at_index(t.input(), ind);
}
template <class ET, class ShapeT, class T, class IndexT>
decltype(auto) element_at_index(reshape_view<ET, ShapeT, T> &t,
                                const IndexT &ind) {
  assert(is_between(ind, 0, (IndexT)t.numel()));
  return element_at_index(t.input(), ind);
}

// reshape
namespace detail {
template <class ET, class OldShapeT, class T, class TT, class ShapeT>
constexpr auto _reshape(const tensor_base<ET, OldShapeT, T> &, TT &&t,
                        const ShapeT &s) {
  return reshape_view<ET, ShapeT, TT>(s, std::forward<TT>(t));
}
template <class ET, class OldShapeT, class T, class TT, class ShapeT>
constexpr auto _reshape(const reshape_view<ET, OldShapeT, T> &, TT &&t,
                        const ShapeT &s) {
  return _reshape(t.input(), std::forward<TT>(t).input(), s);
}
}

// promote
namespace detail {
template <class ET, class ST, class... SizeTs, class T, class TT, class K,
          K Times>
constexpr auto _promote(const tensor_base<ET, tensor_shape<ST, SizeTs...>, T> &,
                        TT &&t, const const_ints<K, Times> &) {
  return reshape(
      std::forward<TT>(t),
      cat2(t.shape(), repeat_shape(const_ints<ST, 1>(), const_size<Times>())));
}
template <class ET, class ST, class... SizeTs, class T, class TT, class K,
          K Times>
constexpr auto _promote(const const_ints<K, Times> &,
                        const tensor_base<ET, tensor_shape<ST, SizeTs...>, T> &,
                        TT &&t) {
  return reshape(
      std::forward<TT>(t),
      cat2(repeat_shape(const_ints<ST, 1>(), const_size<Times>()), t.shape()));
}
}
}