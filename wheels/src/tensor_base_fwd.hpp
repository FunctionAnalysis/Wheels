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

#include <iostream>

#include "shape_fwd.hpp"

namespace wheels {
template <class T> struct tensor_core;

template <class T> struct tensor_iterator;

template <class T1, class T2>
constexpr bool operator==(const tensor_core<T1> &a, const tensor_core<T2> &b);
template <class T1, class T2>
constexpr bool operator!=(const tensor_core<T1> &a, const tensor_core<T2> &b);

template <class ET, class ShapeT, class T> class tensor_base;

// -- necessary tensor functions
// Shape shape_of(ts);
template <class T>
constexpr tensor_shape<size_t> shape_of(const tensor_core<T> &);

// Scalar element_at(ts, subs ...);
template <class T, class... SubTs>
constexpr double element_at(const tensor_core<T> &t, const SubTs &...);
template <class T, class... SubTs>
inline double &element_at(tensor_core<T> &t, const SubTs &...);

// -- auxiliary tensor functions
// auto size_at(ts, const_int);
template <class T, class ST, ST Idx>
constexpr auto size_at(const tensor_core<T> &t, const const_ints<ST, Idx> &idx);

// auto numel_of(ts)
template <class T> constexpr auto numel_of(const tensor_core<T> &t);

// Scalar element_at_index(ts, index);
template <class T, class IndexT>
constexpr decltype(auto) element_at_index(const tensor_core<T> &t,
                                          const IndexT &ind);
template <class T, class IndexT>
decltype(auto) element_at_index(tensor_core<T> &t, const IndexT &ind);

// void reserve_shape(ts, shape);
template <class T, class ST, class... SizeTs>
void reserve_shape(tensor_core<T> &, const tensor_shape<ST, SizeTs...> &shape);

// behavior_flag used in for_each_element*
enum behavior_flag_enum {
  index_ascending,
  unordered,
  break_on_false,
  nonzero_only
};
template <behavior_flag_enum O>
using behavior_flag = const_ints<behavior_flag_enum, O>;

// index_ascending
template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<index_ascending>, FunT fun,
                      const tensor_core<T> &t, Ts &&... ts);

template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<index_ascending>, FunT fun,
                      tensor_core<T> &t, Ts &&... ts);

// unordered
template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<unordered>, FunT fun,
                      const tensor_core<T> &t, Ts &&... ts);

template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<unordered>, FunT fun, tensor_core<T> &t,
                      Ts &&... ts);

// break_on_false
template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<break_on_false>, FunT fun,
                      const tensor_core<T> &t, Ts &&... ts);
template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<break_on_false>, FunT fun,
                      tensor_core<T> &t, Ts &&... ts);
// nonzero_only
template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<nonzero_only>, FunT fun,
                      const tensor_core<T> &t, Ts &&... ts);

template <class FunT, class T, class... Ts>
bool for_each_element(behavior_flag<nonzero_only>, FunT fun, tensor_core<T> &t,
                      Ts &&... ts);

// void assign_elements(to, from);
template <class ToT, class FromT>
void assign_elements(tensor_core<ToT> &to, const tensor_core<FromT> &from);

// void fill_elements_with(to, scalar)
template <class T, class E>
void fill_elements_with(tensor_core<T> &t, const E &e);

// size_t nonzero_elements_count(t)
template <class T> size_t nonzero_elements_count(const tensor_core<T> &t);

// Scalar reduce_elements(ts, initial, functor);
template <class T, class E, class ReduceT>
E reduce_elements(const tensor_core<T> &t, E initial, ReduceT &&red);

// Scalar norm_squared(ts)
template <class ET, class ShapeT, class T>
ET norm_squared(const tensor_base<ET, ShapeT, T> &t);

// Scalar norm_of(ts)
template <class ET, class ShapeT, class T>
constexpr auto norm_of(const tensor_base<ET, ShapeT, T> &t);
// bool all(s)
template <class ET, class ShapeT, class T>
constexpr bool all_of(const tensor_base<ET, ShapeT, T> &t);

// bool any(s)
template <class ET, class ShapeT, class T>
constexpr bool any_of(const tensor_base<ET, ShapeT, T> &t);

// Scalar sum(s)
template <class ET, class ShapeT, class T>
ET sum_of(const tensor_base<ET, ShapeT, T> &t);

// ostream
template <class ET, class ShapeT, class T>
inline std::ostream &operator<<(std::ostream &os,
                                const tensor_base<ET, ShapeT, T> &t);

// is_zero
template <class ET, class ShapeT, class T>
bool is_zero(const tensor_base<ET, ShapeT, T> &t);
}
