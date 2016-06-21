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

#include "block_fwd.hpp"
#include "ewise_fwd.hpp"

#include "tensor_view_base.hpp"

namespace wheels {

template <class ET, class ShapeT, class InputTensorT,
          class... SubscriptTensorTs>
class block_view
    : public tensor_view_base<
          ET, ShapeT,
          block_view<ET, ShapeT, InputTensorT, SubscriptTensorTs...>, false> {
  using _base_t = tensor_view_base<
      ET, ShapeT, block_view<ET, ShapeT, InputTensorT, SubscriptTensorTs...>,
      false>;

public:
  constexpr block_view(InputTensorT &&in, SubscriptTensorTs &&... subts)
      : input_tensor(std::forward<InputTensorT>(in)),
        subs_tensors(std::forward<SubscriptTensorTs>(subts)...) {}

  using _base_t::operator=;
  using _base_t::operator+=;
  using _base_t::operator-=;
  using _base_t::operator*=;
  using _base_t::operator/=;

public:
  InputTensorT input_tensor;
  std::tuple<SubscriptTensorTs...> subs_tensors;
};

// shape_of
namespace details {
template <class SubsViewT, size_t... Is>
constexpr auto _shape_of_block_view(SubsViewT &sv,
                                    const const_ints<size_t, Is...> &) {
  return make_shape(numel_of(std::get<Is>(sv.subs_tensors).derived())...);
}
}
template <class ET, class ShapeT, class InputTensorT,
          class... SubscriptTensorTs>
constexpr auto
shape_of(const block_view<ET, ShapeT, InputTensorT, SubscriptTensorTs...> &t) {
  return details::_shape_of_block_view(
      t, make_const_sequence_for<SubscriptTensorTs...>());
}

// element_at
namespace details {
template <class SubsViewT, class SubsTupleT, size_t... Is>
constexpr decltype(auto)
_element_at_subscript_view_seq(SubsViewT &&sv, SubsTupleT &&subs,
                               const const_ints<size_t, Is...> &) {
  return element_at(
      sv.input_tensor,
      element_at_index(std::get<Is>(sv.subs_tensors), std::get<Is>(subs))...);
}
}
template <class ET, class ShapeT, class InputTensorT,
          class... SubscriptTensorTs, class... SubTs>
constexpr decltype(auto)
element_at(const block_view<ET, ShapeT, InputTensorT, SubscriptTensorTs...> &t,
           const SubTs &... subs) {
  return details::_element_at_subscript_view_seq(
      t, std::forward_as_tuple(subs...),
      make_const_sequence_for<SubscriptTensorTs...>());
}
template <class ET, class ShapeT, class InputTensorT,
          class... SubscriptTensorTs, class... SubTs>
decltype(auto)
element_at(block_view<ET, ShapeT, InputTensorT, SubscriptTensorTs...> &t,
           const SubTs &... subs) {
  return details::_element_at_subscript_view_seq(
      t, std::forward_as_tuple(subs...),
      make_const_sequence_for<SubscriptTensorTs...>());
}

// at_block
namespace details {
template <class InET, class InShapeT, class InT, class InTT,
          class... SubsTensorTs>
constexpr auto _at_block(const tensor_base<InET, InShapeT, InT> &, InTT &&in,
                         SubsTensorTs &&... sts) {
  using shape_t = std::decay_t<decltype(make_shape(sts.numel()...))>;
  return block_view<InET, shape_t, InTT, SubsTensorTs...>(
      std::forward<InTT>(in), std::forward<SubsTensorTs>(sts)...);
}
}
}
