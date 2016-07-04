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
template <class ET, class ShapeT, class T> class reshape_view;

namespace detail {
template <class ET, class OldShapeT, class T, class TT, class ShapeT>
constexpr auto _reshape(const tensor_base<ET, OldShapeT, T> &, TT &&t,
                        const ShapeT &s);
template <class ET, class OldShapeT, class T, class TT, class ShapeT>
constexpr auto _reshape(const reshape_view<ET, OldShapeT, T> &, TT &&t,
                        const ShapeT &s);
}
// reshape
template <class T, class ST, class... SizeTs>
constexpr auto reshape(T &&t, const tensor_shape<ST, SizeTs...> &s)
    -> decltype(detail::_reshape(t, std::forward<T>(t), s)) {
  return detail::_reshape(t, std::forward<T>(t), s);
}

namespace detail {
template <class ET, class ST, class... SizeTs, class T, class TT, class K,
          K Times>
constexpr auto _promote(const tensor_base<ET, tensor_shape<ST, SizeTs...>, T> &,
                        TT &&t, const const_ints<K, Times> &);
template <class ET, class ST, class... SizeTs, class T, class TT, class K,
          K Times>
constexpr auto _promote(const const_ints<K, Times> &,
                        const tensor_base<ET, tensor_shape<ST, SizeTs...>, T> &,
                        TT &&t);
}
// promote
template <class T, class K, K Times>
constexpr auto promote(T &&t, const const_ints<K, Times> &rank)
    -> decltype(detail::_promote(t, std::forward<T>(t), rank)) {
  return detail::_promote(t, std::forward<T>(t), rank);
}
template <class T, class K, K Times>
constexpr auto promote(const const_ints<K, Times> &rank, T &&t)
    -> decltype(detail::_promote(rank, t, std::forward<T>(t))) {
  return detail::_promote(rank, t, std::forward<T>(t));
}
}
