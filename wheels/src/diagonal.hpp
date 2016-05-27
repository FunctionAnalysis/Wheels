#pragma once

#include "tensor_base.hpp"
#include "constants.hpp"

#include "diagonal_fwd.hpp"

namespace wheels {

// make_diag_result
template <class ET, class ShapeT, class T>
class make_diag_result
    : public tensor_base<ET, ShapeT, make_diag_result<ET, ShapeT, T>> {
public:
  constexpr explicit make_diag_result(const ShapeT &s, T &&in)
      : _shape(s), _input(std::forward<T>(in)) {
    assert(_input.numel() == min_shape_size(_shape));
  }
  constexpr const ShapeT &shape() const { return _shape; }
  constexpr const T &input() const & { return _input; }
  T &input() & { return _input; }
  T &&input() && { return std::move(_input); }

private:
  ShapeT _shape;
  T _input;
};

// shape_of
template <class ET, class ShapeT, class T>
constexpr decltype(auto) shape_of(const make_diag_result<ET, ShapeT, T> &t) {
  return t.shape();
}

// element_at
template <class ET, class ShapeT, class T, class SubT, class... SubTs>
constexpr decltype(auto) element_at(const make_diag_result<ET, ShapeT, T> &t,
                                    const SubT &sub, const SubTs &... subs) {
  return conditional(all_same(sub, subs...), element_at_index(t.input(), sub),
                     types<ET>::zero());
}

// for_each(nonzero)
template <class FunT, class ET, class ShapeT, class T>
bool for_each_element(behavior_flag<nonzero_only> o, FunT fun,
                      const make_diag_result<ET, ShapeT, T> &r) {
  return for_each_element(o, fun, r.input());
}

// sum_of
template <class ET, class ShapeT, class T>
constexpr auto sum_of(const make_diag_result<ET, ShapeT, T> &r) {
  return sum_of(r.input());
}

// norm_squared
template <class ET, class ShapeT, class T>
constexpr auto norm_squared(const make_diag_result<ET, ShapeT, T> &r) {
  return norm_squared(r.input());
}

// all_of
template <class ET, class ShapeT, class T>
constexpr bool all_of(const make_diag_result<ET, ShapeT, T> &r) {
  return conditional(r.numel() == const_int<1>(), all_of(r.input()), false);
}

// any_of
template <class ET, class ShapeT, class T>
constexpr bool any_of(const make_diag_result<ET, ShapeT, T> &r) {
  return any_of(r.input());
}

// diag_view
template <class ET, class ShapeT, class T>
class diag_view : public tensor_base<ET, ShapeT, diag_view<ET, ShapeT, T>> {
public:
  constexpr explicit diag_view(T &&in) : _input(std::forward<T>(in)) {}
  constexpr auto shape() const {
    return make_shape(min_shape_size(_input.shape()));
  }
  constexpr const T &input() const & { return _input; }
  T &input() & { return _input; }
  T &&input() && { return _input; }

  // operator=
  template <class AnotherT>
  diag_view &operator=(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
    return *this;
  }
  diag_view &operator=(const ET &e) {
    fill_elements_with(*this, e);
    return *this;
  }

private:
  T _input;
};

// shape_of
template <class ET, class ShapeT, class T>
constexpr decltype(auto) shape_of(const diag_view<ET, ShapeT, T> &t) {
  return t.shape();
}

// element_at_index
namespace details {
template <class DiagViewT, class IndexT, size_t... Is>
constexpr decltype(auto)
_element_at_index_diagview(DiagViewT &t, const IndexT &ind,
                           const const_ints<size_t, Is...> &) {
  return element_at(t.input(), always_f<IndexT>(ind)(Is)...);
}
}
template <class ET, class ShapeT, class T, class IndexT>
constexpr decltype(auto) element_at_index(const diag_view<ET, ShapeT, T> &t,
                                          const IndexT &ind) {
  return details::_element_at_index_diagview(
      t, ind, make_rank_sequence(t.input().shape()));
}
template <class ET, class ShapeT, class T, class IndexT>
decltype(auto) element_at_index(diag_view<ET, ShapeT, T> &t,
                                const IndexT &ind) {
  return details::_element_at_index_diagview(
      t, ind, make_rank_sequence(t.input().shape()));
}

// element_at
template <class ET, class ShapeT, class T, class SubT>
constexpr decltype(auto) element_at(const diag_view<ET, ShapeT, T> &t,
                                    const SubT &ind) {
  return details::_element_at_index_diagview(
      t, ind, make_rank_sequence(t.input().shape()));
}
template <class ET, class ShapeT, class T, class SubT>
decltype(auto) element_at(diag_view<ET, ShapeT, T> &t, const SubT &ind) {
  return details::_element_at_index_diagview(
      t, ind, make_rank_sequence(t.input().shape()));
}

// make_diag
namespace details {
template <class ET, class ShapeT, class T, class TT, class NewShapeT>
constexpr auto _make_diag(const tensor_base<ET, ShapeT, T> &, TT &&t,
                          const NewShapeT &nshape) {
  assert(t.numel() == min_shape_size(nshape));
  return make_diag_result<ET, NewShapeT, TT>(nshape, std::forward<TT>(t));
}
}

// eye
template <class ET, class ST, class... SizeTs>
constexpr auto eye(const tensor_shape<ST, SizeTs...> &s) {
  return make_diag(ones<ET>(make_shape(min_shape_size(s))), s);
}
template <class ET, class MT, class NT>
constexpr auto eye(const MT &m, const NT &n) {
  return make_diag(ones<ET>(make_shape(min(m, n))), make_shape(m, n));
}
template <class ET, class NT, class> constexpr auto eye(const NT &n) {
  return make_diag(ones<ET>(make_shape(n)), make_shape(n, n));
}

// diag
namespace details {
template <class ET, class ShapeT, class T, class TT>
constexpr auto _diag(const tensor_base<ET, ShapeT, T> &, TT &&t) {
  using shape_t = decltype(make_shape(min_shape_size(t.shape())));
  return diag_view<ET, shape_t, TT>(std::forward<TT>(t));
}
template <class ET, class ShapeT, class T, class TT>
constexpr decltype(auto) _diag(const make_diag_result<ET, ShapeT, T> &,
                               TT &&t) {
  return std::forward<TT>(t).input();
}
}
}
