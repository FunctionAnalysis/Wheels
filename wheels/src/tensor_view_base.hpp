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

#include "aligned.hpp"
#include "ewise.hpp"
#include "tensor_base.hpp"

namespace wheels {
namespace details {
template <class ET, class ShapeT, class T, bool IsContinuousData>
struct _select_tensor_base;

template <class ET, class ShapeT, class T>
struct _select_tensor_base<ET, ShapeT, T, false> {
  using type = tensor_base<ET, ShapeT, T>;
};
template <class ET, class ShapeT, class T>
struct _select_tensor_base<ET, ShapeT, T, true> {
  using type = tensor_continuous_data_base<ET, ShapeT, T>;
};
template <class ET, class ShapeT, class T, bool IsContinuousData>
using _select_tensor_base_t =
    typename _select_tensor_base<ET, ShapeT, T, IsContinuousData>::type;
}

template <class ET, class ShapeT, class T, bool IsContinuousData = false>
class tensor_view_base
    : public details::_select_tensor_base_t<ET, ShapeT, T, IsContinuousData> {
public:
  // operator=
  template <class AnotherShapeT, class AnotherT,
            class = std::enable_if_t<AnotherShapeT::rank == ShapeT::rank>>
  T &operator=(const tensor_base<ET, AnotherShapeT, AnotherT> &another) {
    assign_elements(this->derived(), another.derived());
    return this->derived();
  }
  T &operator=(const ET &e) {
    fill_elements_with(this->derived(), e);
    return this->derived();
  }
  template <class AnotherET, class AnotherShapeT, class AnotherT>
  T &operator=(const scalarize_wrapper<AnotherET, AnotherShapeT, AnotherT> &t) {
    fill_elements_with(this->derived(), t.host);
    return this->derived();
  }

  // +=
  template <class AnotherShapeT, class AnotherT,
            class = std::enable_if_t<AnotherShapeT::rank == ShapeT::rank>>
  T &operator+=(const tensor_base<ET, AnotherShapeT, AnotherT> &t) {
    assert(this->shape() == t.shape());
    for_each_element(behavior_flag<unordered>(),
                     [](auto &&ele1, auto &&ele2) { ele1 += ele2; },
                     this->derived(), t.derived());
    return this->derived();
  }
  T &operator+=(const ET &e) {
    for_each_element(behavior_flag<unordered>(), [&e](auto &&ele) { ele += e; },
                     this->derived());
    return this->derived();
  }
  template <class AnotherET, class AnotherShapeT, class AnotherT>
  T &operator+=(
      const scalarize_wrapper<AnotherET, AnotherShapeT, AnotherT> &t) {
    for_each_element(behavior_flag<unordered>(),
                     [&t](auto &&ele) { ele += t.host; }, this->derived());
    return this->derived();
  }

  // -=
  template <class AnotherShapeT, class AnotherT,
            class = std::enable_if_t<AnotherShapeT::rank == ShapeT::rank>>
  T &operator-=(const tensor_base<ET, AnotherShapeT, AnotherT> &t) {
    assert(this->shape() == t.shape());
    for_each_element(behavior_flag<unordered>(),
                     [](auto &&ele1, auto &&ele2) { ele1 -= ele2; },
                     this->derived(), t.derived());
    return this->derived();
  }
  T &operator-=(const ET &e) {
    for_each_element(behavior_flag<unordered>(), [&e](auto &&ele) { ele -= e; },
                     this->derived());
    return this->derived();
  }
  template <class AnotherET, class AnotherShapeT, class AnotherT>
  T &operator-=(
      const scalarize_wrapper<AnotherET, AnotherShapeT, AnotherT> &t) {
    for_each_element(behavior_flag<unordered>(),
                     [&t](auto &&ele) { ele -= t.host; }, this->derived());
    return this->derived();
  }

  // *=
  T &operator*=(const ET &e) {
    for_each_element(behavior_flag<unordered>(), [&e](auto &&ele) { ele *= e; },
                     this->derived());
    return this->derived();
  }

  // /=
  T &operator/=(const ET &e) {
    for_each_element(behavior_flag<unordered>(), [&e](auto &&ele) { ele /= e; },
                     this->derived());
    return this->derived();
  }
};
}