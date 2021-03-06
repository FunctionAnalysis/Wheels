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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

#include "const_ints.hpp"

#include "tensor_base.hpp"

#include "constants_fwd.hpp"

namespace wheels {

// constant_result
template <class ET, class ShapeT>
class constant_result
    : public tensor_base<ET, ShapeT, constant_result<ET, ShapeT>> {
public:
  using shape_type = ShapeT;
  using value_type = ET;
  template <class EE>
  constexpr explicit constant_result(const ShapeT &s, EE &&v)
      : _shape(s), _val(std::forward<EE>(v)) {}
  const ShapeT &shape() const { return _shape; }
  const ET &value() const { return _val; }
  ET &value() { return _val; }

private:
  ShapeT _shape;
  ET _val;
};

// shape_of
template <class ET, class ShapeT>
constexpr const ShapeT &shape_of(const constant_result<ET, ShapeT> &t) {
  return t.shape();
}

// element_at
template <class ET, class ShapeT, class... SubTs>
constexpr const ET &element_at(const constant_result<ET, ShapeT> &t,
                               const SubTs &... subs) {
  return t.value();
}

// element_at_index
template <class ET, class ShapeT, class IndexT>
constexpr const ET &element_at_index(const constant_result<ET, ShapeT> &t,
                                     const IndexT &ind) {
  assert(is_between(ind, 0, (IndexT)t.numel()));
  return t.value();
}

// index_ascending, unordered
template <behavior_flag_enum O, class FunT, class ET, class ShapeT>
bool for_each_element(behavior_flag<O>, FunT fun,
                      const constant_result<ET, ShapeT> &t) {
  for (size_t i = 0; i < numel_of(t); i++) {
    fun(t.value());
  }
  return true;
}

// break_on_false
template <class FunT, class ET, class ShapeT>
bool for_each_element(behavior_flag<break_on_false>, FunT fun,
                      const constant_result<ET, ShapeT> &t) {
  for (size_t i = 0; i < numel_of(t); i++) {
    if (!fun(t.value())) {
      return false;
    }
  }
  return true;
}

// nonzero_only
template <class FunT, class ET, class ShapeT, class... Ts>
bool for_each_element(behavior_flag<nonzero_only> o, FunT fun,
                      const constant_result<ET, ShapeT> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  if (!is_zero(t.value())) {
    return for_each_element(behavior_flag<unordered>(), std::forward<FunT>(fun),
                            t, std::forward<Ts>(ts)...);
  }
  return false;
}

// size_t nonzero_elements_count(t)
template <class ET, class ShapeT>
size_t nonzero_elements_count(const constant_result<ET, ShapeT> &t) {
  return is_zero(t.value()) ? 0 : numel_of(t);
}

// reduce_elements
template <class ET, class ShapeT, class E, class ReduceT>
E reduce_elements(const constant_result<ET, ShapeT> &t, E initial,
                  ReduceT &&red) {
  for (size_t i = 0; i < numel_of(t); i++) {
    initial = red(initial, t.value());
  }
  return initial;
}

// sum_of
template <class ET, class ShapeT>
std::enable_if_t<std::is_scalar<ET>::value, ET>
sum_of(const constant_result<ET, ShapeT> &t) {
  return t.value() * numel_of(t);
}

// norm_squared
template <class ET, class ShapeT>
std::enable_if_t<std::is_scalar<ET>::value, ET>
norm_squared(const constant_result<ET, ShapeT> &t) {
  return t.value() * t.value() * numel_of(t);
}

// all_of
template <class ET, class ShapeT>
constexpr bool all_of(const constant_result<ET, ShapeT> &t) {
  return !!t.value();
}

// any_of
template <class ET, class ShapeT>
constexpr bool any_of(const constant_result<ET, ShapeT> &t) {
  return !!t.value();
}

// constants
template <class ET, class ST, class... SizeTs>
constexpr auto constants(const tensor_shape<ST, SizeTs...> &shape, ET &&v) {
  return constant_result<std::decay_t<ET>, tensor_shape<ST, SizeTs...>>(
      shape, std::forward<ET>(v));
}

// zeros
template <class ET, class ST, class... SizeTs>
constexpr auto zeros(const tensor_shape<ST, SizeTs...> &shape) {
  return constants<ET>(shape, 0);
}
template <class ET, class... SizeTs>
constexpr auto zeros(const SizeTs &... sizes) {
  return constants<ET>(make_shape(sizes...), 0);
}

// ones
template <class ET, class ST, class... SizeTs>
constexpr auto ones(const tensor_shape<ST, SizeTs...> &shape) {
  return constants<ET>(shape, 1);
}
template <class ET, class... SizeTs>
constexpr auto ones(const SizeTs &... sizes) {
  return constants<ET>(make_shape(sizes...), 1);
}
}