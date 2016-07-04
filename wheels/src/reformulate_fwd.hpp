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

#include "tensor_base_fwd.hpp"

namespace wheels {

template <class ET, class ShapeT, class InputT, class SubsMapFunT>
class reformulate_result;

namespace detail {
template <class ET, class ShapeT, class T, class TT, class NewShapeT,
          class SubsMapFunT>
constexpr auto _reformulate(const tensor_base<ET, ShapeT, T> &, TT &&t,
                            const NewShapeT &s, SubsMapFunT fun);
}
// reformulate
template <class T, class ST, class... SizeTs, class SubsMapFunT>
constexpr auto reformulate(T &&t, const tensor_shape<ST, SizeTs...> &shape,
                           SubsMapFunT fun)
    -> decltype(detail::_reformulate(t, std::forward<T>(t), shape, fun)) {
  return detail::_reformulate(t, std::forward<T>(t), shape, fun);
}

namespace detail {
template <class ET, class ShapeT, class T, class TT, class RepsTupleT,
          size_t... Is>
constexpr auto _repeat_impl(const tensor_base<ET, ShapeT, T> &, TT &&t,
                            RepsTupleT &&rps,
                            const const_ints<size_t, Is...> &);
}

// repeat
template <class T, class... RepTs>
constexpr auto repeat(T &&t, const RepTs &... reps)
    -> decltype(detail::_repeat_impl(t, std::forward<T>(t),
                                      std::forward_as_tuple(reps...),
                                      make_const_sequence_for<RepTs...>())) {
  return detail::_repeat_impl(t, std::forward<T>(t),
                               std::forward_as_tuple(reps...),
                               make_const_sequence_for<RepTs...>());
}
}