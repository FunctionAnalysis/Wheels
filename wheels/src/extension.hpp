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

#include "extension_fwd.hpp"

namespace wheels {

template <class ExtensionT, class EleT, class ShapeT, class T>
class tensor_extension_base : public tensor_base<EleT, ShapeT, T> {};

template <class ExtensionT, class EleT, class ShapeT, class T>
class tensor_extension_wrapper
    : public tensor_extension_base<
          ExtensionT, EleT, ShapeT,
          tensor_extension_wrapper<ExtensionT, EleT, ShapeT, T>> {
public:
  explicit tensor_extension_wrapper(T &&h) : host(std::forward<T>(h)) {}

public:
  T host;
};

namespace detail {
template <class ExtensionT, class EleT, class ShapeT, class T, class TT>
constexpr auto _extend(const tensor_base<EleT, ShapeT, T> &, TT &&host) {
  return tensor_extension_wrapper<ExtensionT, EleT, ShapeT, TT>(
      std::forward<TT>(host));
}
}

// -- necessary tensor functions
// Shape shape_of(ts);
template <class ExtensionT, class EleT, class ShapeT, class T>
constexpr decltype(auto)
shape_of(const tensor_extension_wrapper<ExtensionT, EleT, ShapeT, T> &t) {
  return shape_of(t.host);
}

// Scalar element_at(ts, subs ...);
template <class ExtensionT, class EleT, class ShapeT, class T, class... SubTs>
constexpr decltype(auto)
element_at(const tensor_extension_wrapper<ExtensionT, EleT, ShapeT, T> &t,
           const SubTs &... subs) {
  assert(subscripts_are_valid(t.shape(), subs...));
  return element_at(t.host, subs...);
}
template <class ExtensionT, class EleT, class ShapeT, class T, class... SubTs>
decltype(auto)
element_at(tensor_extension_wrapper<ExtensionT, EleT, ShapeT, T> &t,
           const SubTs &... subs) {
  assert(subscripts_are_valid(t.shape(), subs...));
  return element_at(t.host, subs...);
}

// Scalar element_at_index(ts, index);
template <class ExtensionT, class EleT, class ShapeT, class T, class IndexT>
constexpr decltype(auto)
element_at_index(const tensor_extension_wrapper<ExtensionT, EleT, ShapeT, T> &t,
                 const IndexT &ind) {
  assert(is_between(ind, 0, (IndexT)t.numel()));
  return element_at_index(t.host, ind);
}
template <class ExtensionT, class EleT, class ShapeT, class T, class IndexT>
decltype(auto)
element_at_index(tensor_extension_wrapper<ExtensionT, EleT, ShapeT, T> &t,
                 const IndexT &ind) {
  assert(is_between(ind, 0, (IndexT)t.numel()));
  return element_at_index(t.host, ind);
}

// void reserve_shape(ts, shape);
template <class ExtensionT, class EleT, class ShapeT, class T, class ST,
          class... SizeTs>
void reserve_shape(tensor_extension_wrapper<ExtensionT, EleT, ShapeT, T> &t,
                   const tensor_shape<ST, SizeTs...> &shape) {
  reserve_shape(t.host, shape);
}

// for_each_element
template <behavior_flag_enum F, class FunT, class ExtensionT, class EleT,
          class ShapeT, class T, class... Ts>
constexpr bool
for_each_element(behavior_flag<F> f, FunT fun,
                 const tensor_extension_wrapper<ExtensionT, EleT, ShapeT, T> &t,
                 Ts &... ts) {
  return for_each_element(f, fun, t.host, ts...);
}

// fill_elements_with
template <class ExtensionT, class EleT, class ShapeT, class T, class E>
void fill_elements_with(
    tensor_extension_wrapper<ExtensionT, EleT, ShapeT, T> &t, const E &e) {
  fill_elements_with(t.host, e);
}

// size_t nonzero_elements_count(t)
template <class ExtensionT, class EleT, class ShapeT, class T>
constexpr size_t nonzero_elements_count(
    const tensor_extension_wrapper<ExtensionT, EleT, ShapeT, T> &t) {
  return nonzero_elements_count(t.host);
}

// Scalar reduce_elements(ts, initial, functor);
template <class ExtensionT, class EleT, class ShapeT, class T, class E,
          class ReduceT>
constexpr E
reduce_elements(const tensor_extension_wrapper<ExtensionT, EleT, ShapeT, T> &t,
                E initial, ReduceT &red) {
  return reduce_elements(t.host, initial, red);
}

// Scalar norm_squared(ts)
template <class ExtensionT, class ET, class ShapeT, class T>
constexpr ET
norm_squared(const tensor_extension_wrapper<ExtensionT, ET, ShapeT, T> &t) {
  return norm_squared(t.host);
}

// Scalar norm_of(ts)
template <class ExtensionT, class ET, class ShapeT, class T>
constexpr auto
norm_of(const tensor_extension_wrapper<ExtensionT, ET, ShapeT, T> &t) {
  return norm_of(t.host);
}

// bool all(s)
template <class ExtensionT, class ET, class ShapeT, class T>
constexpr bool
all_of(const tensor_extension_wrapper<ExtensionT, ET, ShapeT, T> &t) {
  return all_of(t.host);
}

// bool any(s)
template <class ExtensionT, class ET, class ShapeT, class T>
constexpr bool
any_of(const tensor_extension_wrapper<ExtensionT, ET, ShapeT, T> &t) {
  return any_of(t.host);
}

// Scalar sum(s)
template <class ExtensionT, class ET, class ShapeT, class T>
ET sum_of(const tensor_extension_wrapper<ExtensionT, ET, ShapeT, T> &t) {
  return sum_of(t.host);
}
}
