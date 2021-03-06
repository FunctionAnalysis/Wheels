/* * *
 * The MIT License (MIT)
 * 
 * Copyright (c) 2016 Hao Yang (yangh2007@gmail.com)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * * */

#pragma once

#include "tensor_base.hpp"
#include "tensor_view_base.hpp"
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
  assert(subscripts_are_valid(t.shape(), sub, subs...));
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
class diag_view
    : public tensor_view_base<ET, ShapeT, diag_view<ET, ShapeT, T>, false> {
  using _base_t = tensor_view_base<ET, ShapeT, diag_view<ET, ShapeT, T>, false>;

public:
  constexpr explicit diag_view(T &&in) : _input(std::forward<T>(in)) {}
  constexpr auto shape() const {
    return make_shape(min_shape_size(_input.shape()));
  }
  constexpr const T &input() const & { return _input; }
  T &input() & { return _input; }
  T &&input() && { return _input; }

  using _base_t::operator=;
  using _base_t::operator+=;
  using _base_t::operator-=;
  using _base_t::operator*=;
  using _base_t::operator/=;

private:
  T _input;
};

// shape_of
template <class ET, class ShapeT, class T>
constexpr decltype(auto) shape_of(const diag_view<ET, ShapeT, T> &t) {
  return t.shape();
}

// element_at_index
namespace detail {
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
  assert(is_between(ind, 0, (IndexT)t.numel()));
  return detail::_element_at_index_diagview(
      t, ind, make_rank_sequence(t.input().shape()));
}
template <class ET, class ShapeT, class T, class IndexT>
decltype(auto) element_at_index(diag_view<ET, ShapeT, T> &t,
                                const IndexT &ind) {
  assert(is_between(ind, 0, (IndexT)t.numel()));
  return detail::_element_at_index_diagview(
      t, ind, make_rank_sequence(t.input().shape()));
}

// element_at
template <class ET, class ShapeT, class T, class SubT>
constexpr decltype(auto) element_at(const diag_view<ET, ShapeT, T> &t,
                                    const SubT &ind) {
  assert(subscripts_are_valid(t.shape(), ind));
  return detail::_element_at_index_diagview(
      t, ind, make_rank_sequence(t.input().shape()));
}
template <class ET, class ShapeT, class T, class SubT>
decltype(auto) element_at(diag_view<ET, ShapeT, T> &t, const SubT &ind) {
  assert(subscripts_are_valid(t.shape(), ind));
  return detail::_element_at_index_diagview(
      t, ind, make_rank_sequence(t.input().shape()));
}

// make_diag
namespace detail {
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
namespace detail {
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
