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
#include "tensor.hpp"

#include "downgrade_fwd.hpp"
#include "upgrade_fwd.hpp"

namespace wheels {

namespace detail {
template <class T1, class T2>
struct _is_same_intrinsic : std::is_same<std::decay_t<T1>, std::decay_t<T2>> {};
}

// subtensor_view
template <class ET, class SubShapeT, class InputT, size_t FixedRank>
class subtensor_view
    : public tensor_view_base<ET, SubShapeT,
                              subtensor_view<ET, SubShapeT, InputT, FixedRank>,
                              false> {
  using _base_t =
      tensor_view_base<ET, SubShapeT,
                       subtensor_view<ET, SubShapeT, InputT, FixedRank>, false>;

public:
  template <class... SubTs>
  constexpr subtensor_view(InputT &&in, const SubTs &... subs)
      : input(std::forward<InputT>(in)), fixed_subs{{(size_t)subs...}} {}

  using _base_t::operator=;
  using _base_t::operator+=;
  using _base_t::operator-=;
  using _base_t::operator*=;
  using _base_t::operator/=;

public:
  InputT input;
  std::array<size_t, FixedRank> fixed_subs;
};

// shape_of
template <class ET, class ShapeT, class InputT, size_t FixedRank>
constexpr auto
shape_of(const subtensor_view<ET, ShapeT, InputT, FixedRank> &b) {
  return b.input.shape().part(make_const_range(
      const_index<FixedRank>(), const_index<FixedRank + ShapeT::rank>()));
}

// element_at
namespace detail {
template <class SubTensorViewT, size_t... Is, class... SubTs>
constexpr decltype(auto)
_element_at_subtensor_view_seq(SubTensorViewT &b,
                               const const_ints<size_t, Is...> &,
                               const SubTs &... subs) {
  return element_at(b.input, std::get<Is>(b.fixed_subs)..., subs...);
}
}
template <class ET, class ShapeT, class InputT, size_t FixedRank,
          class... SubTs>
constexpr decltype(auto)
element_at(const subtensor_view<ET, ShapeT, InputT, FixedRank> &b,
           const SubTs &... subs) {
  assert(subscripts_are_valid(b.shape(), subs...));
  return detail::_element_at_subtensor_view_seq(
      b, make_const_sequence(const_size<FixedRank>()), subs...);
}
template <class ET, class ShapeT, class InputT, size_t FixedRank,
          class... SubTs>
decltype(auto) element_at(subtensor_view<ET, ShapeT, InputT, FixedRank> &b,
                          const SubTs &... subs) {
  assert(subscripts_are_valid(b.shape(), subs...));
  return detail::_element_at_subtensor_view_seq(
      b, make_const_sequence(const_size<FixedRank>()), subs...);
}

// subtensor_at
namespace detail {
// _cat_shape
template <class ShapeT1, class ShapeT2> struct _cat_shape {};
template <class T1, class... SizeT1s, class T2, class... SizeT2s>
struct _cat_shape<tensor_shape<T1, SizeT1s...>, tensor_shape<T2, SizeT2s...>> {
  using type = tensor_shape<std::common_type_t<T1, T2>, SizeT1s..., SizeT2s...>;
};
template <class T, class... SizeTs, class SizeT>
struct _cat_shape<tensor_shape<T, SizeTs...>, SizeT> {
  using type = tensor_shape<T, SizeTs..., SizeT>;
};
template <class T, class... SizeTs, class SizeT>
struct _cat_shape<SizeT, tensor_shape<T, SizeTs...>> {
  using type = tensor_shape<T, SizeT, SizeTs...>;
};

// _split_shape
template <class ShapeT, size_t N> struct _split_shape {};
template <class T, class... SizeTs>
struct _split_shape<tensor_shape<T, SizeTs...>, (size_t)0> {
  using head = tensor_shape<T>;
  using tail = tensor_shape<T, SizeTs...>;
};
template <class T, class SizeT, class... SizeTs>
struct _split_shape<tensor_shape<T, SizeT, SizeTs...>, (size_t)0> {
  using head = tensor_shape<T>;
  using tail = tensor_shape<T, SizeT, SizeTs...>;
};
template <class T, class SizeT, class... SizeTs, size_t N>
struct _split_shape<tensor_shape<T, SizeT, SizeTs...>, N> {
  using head = typename _cat_shape<
      SizeT,
      typename _split_shape<tensor_shape<T, SizeTs...>, N - 1>::head>::type;
  using tail = typename _split_shape<tensor_shape<T, SizeTs...>, N - 1>::tail;
};

template <class ShapeT, size_t N>
using _head_of_shape_t = typename _split_shape<ShapeT, N>::head;
template <class ShapeT, size_t N>
using _tail_of_shape_t = typename _split_shape<ShapeT, N>::tail;

template <class ET, class ShapeT, class T, class TT, class... SubTs>
constexpr auto _subtensor_at(const tensor_base<ET, ShapeT, T> &, TT &&input,
                             const SubTs &... subs) {
  static_assert(sizeof...(SubTs) < ShapeT::rank, "two many subscripts");
  using shape_t = _tail_of_shape_t<ShapeT, sizeof...(SubTs)>;
  return subtensor_view<ET, shape_t, TT, sizeof...(SubTs)>(
      std::forward<TT>(input), subs...);
}
}

// downgrade_view
template <class ET, class ShapeT, class InputT, size_t FixedRank>
class downgrade_view
    : public tensor_view_base<
          tensor<ET, detail::_tail_of_shape_t<ShapeT, FixedRank>>,
          detail::_head_of_shape_t<ShapeT, FixedRank>,
          downgrade_view<ET, ShapeT, InputT, FixedRank>, false> {
  static_assert(FixedRank <= ShapeT::rank, "fixed rank overflow");
  using _base_t =
      tensor_view_base<tensor<ET, detail::_tail_of_shape_t<ShapeT, FixedRank>>,
                       detail::_head_of_shape_t<ShapeT, FixedRank>,
                       downgrade_view<ET, ShapeT, InputT, FixedRank>, false>;

public:
  constexpr explicit downgrade_view(InputT &&in)
      : input(std::forward<InputT>(in)) {}

  using _base_t::operator=;
  using _base_t::operator+=;
  using _base_t::operator-=;
  using _base_t::operator*=;
  using _base_t::operator/=;

public:
  InputT input;
};

// shape_of
template <class ET, class ShapeT, class InputT, size_t FixedRank>
constexpr auto
shape_of(const downgrade_view<ET, ShapeT, InputT, FixedRank> &t) {
  return t.input.shape().part(
      make_const_range(const_index<0>(), const_index<FixedRank>()));
}

// element_at
namespace detail {
template <class SubTensorViewT, class SubwiseViewT, size_t... Is,
          class SubsTupleT>
constexpr SubTensorViewT _element_at_downgrade_view_seq(
    SubwiseViewT &bv, const const_ints<size_t, Is...> &, SubsTupleT &&subs) {
  return SubTensorViewT(bv.input, std::get<Is>(subs)...);
}
}

template <class ET, class ShapeT, class InputT, size_t FixedRank,
          class... SubTs>
constexpr auto
element_at(const downgrade_view<ET, ShapeT, InputT, FixedRank> &t,
           const SubTs &... subs) {
  assert(subscripts_are_valid(t.shape(), subs...));
  using const_subtensor_view_view_t =
      subtensor_view<ET, detail::_tail_of_shape_t<ShapeT, FixedRank>,
                     const InputT &, FixedRank>;
  return detail::_element_at_downgrade_view_seq<const_subtensor_view_view_t>(
      t, make_const_sequence(const_size<FixedRank>()),
      std::forward_as_tuple(subs...));
}

template <class ET, class ShapeT, class InputT, size_t FixedRank,
          class... SubTs>
auto element_at(downgrade_view<ET, ShapeT, InputT, FixedRank> &t,
                const SubTs &... subs) {
  assert(subscripts_are_valid(t.shape(), subs...));
  using subtensor_view_view_t =
      subtensor_view<ET, detail::_tail_of_shape_t<ShapeT, FixedRank>, InputT &,
                     FixedRank>;
  return detail::_element_at_downgrade_view_seq<subtensor_view_view_t>(
      t, make_const_sequence(const_size<FixedRank>()),
      std::forward_as_tuple(subs...));
}

// downgrade
namespace detail {
template <class ET, class ShapeT, class InputT, class TT, size_t FixedRank>
constexpr auto _downgrade(const tensor_base<ET, ShapeT, InputT> &, TT &&input,
                          const const_size<FixedRank> &) {
  return downgrade_view<ET, ShapeT, TT, FixedRank>(std::forward<TT>(input));
}
// downgrade an upgraded tensor
template <class ET, class ShapeT, class InputT, class ExtShapeT, class TT>
constexpr decltype(auto)
_downgrade(const upgrade_result<ET, ShapeT, InputT, ExtShapeT,
                                upgrade_result_ops::as_subtensor> &,
           TT &&input, const const_size<ExtShapeT::rank> &) {
  return std::forward<TT>(input).input;
}
}
}
