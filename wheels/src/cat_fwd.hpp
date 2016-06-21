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

#include "const_ints.hpp"

#include "tensor_base_fwd.hpp"

namespace wheels {

// cat_result
template <class ET, class ShapeT, size_t Axis, class T1, class T2>
class cat_result;

namespace details {
template <size_t Axis, class ShapeT1, class ET1, class T1, class ShapeT2,
          class ET2, class T2, class TT1, class TT2>
constexpr auto _cat_tensor_at(const const_index<Axis> &axis,
                              const tensor_base<ET1, ShapeT1, T1> &,
                              const tensor_base<ET2, ShapeT2, T2> &, TT1 &&in1,
                              TT2 &&in2);
}

template <size_t Axis, class T1, class T2>
constexpr auto cat_at(const const_index<Axis> &axis, T1 &&in1, T2 &&in2)
    -> decltype(details::_cat_tensor_at(axis, in1, in2, std::forward<T1>(in1),
                                        std::forward<T2>(in2))) {
  return details::_cat_tensor_at(axis, in1, in2, std::forward<T1>(in1),
                                 std::forward<T2>(in2));
}

// cat2 (cat_at 0)
template <class T1, class T2>
constexpr auto cat2(T1 &&in1, T2 &&in2)
    -> decltype(cat_at(const_index<0>(), std::forward<T1>(in1),
                       std::forward<T2>(in2))) {
  return cat_at(const_index<0>(), std::forward<T1>(in1), std::forward<T2>(in2));
}

template <class ET, class ShapeT, size_t Axis, class T1, class T2,
          class... SubTs>
constexpr ET element_at(const cat_result<ET, ShapeT, Axis, T1, T2> &m,
                        const SubTs &... subs);

// unordered
template <class FunT, class ET, class ShapeT, size_t Axis, class T1, class T2>
bool for_each_element(behavior_flag<unordered> o, FunT fun,
                      const cat_result<ET, ShapeT, Axis, T1, T2> &t);

// break_on_false
template <class FunT, class ET, class ShapeT, size_t Axis, class T1, class T2>
bool for_each_element(behavior_flag<break_on_false> o, FunT fun,
                      const cat_result<ET, ShapeT, Axis, T1, T2> &t);
// nonzero_only
template <class FunT, class ET, class ShapeT, size_t Axis, class T1, class T2>
bool for_each_element(behavior_flag<nonzero_only> o, FunT fun,
                      const cat_result<ET, ShapeT, Axis, T1, T2> &t);

// size_t nonzero_elements_count(t)
template <class ET, class ShapeT, size_t Axis, class T1, class T2>
inline size_t
nonzero_elements_count(const cat_result<ET, ShapeT, Axis, T1, T2> &t);
}
