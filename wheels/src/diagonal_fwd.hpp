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

#include "shape_fwd.hpp"
#include "tensor_base_fwd.hpp"

namespace wheels {

// make_diag_result
template <class ET, class ShapeT, class T> class make_diag_result;

// diag_view
template <class ET, class ShapeT, class T> class diag_view;

// make_diag
namespace details {
template <class ET, class ShapeT, class T, class TT, class NewShapeT>
constexpr auto _make_diag(const tensor_base<ET, ShapeT, T> &, TT &&t,
                          const NewShapeT &nshape);
}
template <class T, class ST, class... SizeTs>
constexpr auto make_diag(T &&t, const tensor_shape<ST, SizeTs...> &ns)
    -> decltype(details::_make_diag(t, std::forward<T>(t), ns)) {
  return details::_make_diag(t, std::forward<T>(t), ns);
}
template <class T>
constexpr auto make_diag(T &&t)
    -> decltype(details::_make_diag(t, std::forward<T>(t),
                                    make_shape(t.numel(), t.numel()))) {
  return details::_make_diag(t, std::forward<T>(t),
                             make_shape(t.numel(), t.numel()));
}

// eye
template <class ET = double, class ST, class... SizeTs>
constexpr auto eye(const tensor_shape<ST, SizeTs...> &s);
template <class ET = double, class MT, class NT>
constexpr auto eye(const MT &m, const NT &n);
template <class ET = double, class NT,
          class = std::enable_if_t<!is_tensor_shape<NT>::value>>
constexpr auto eye(const NT &n);

// diag
namespace details {
template <class ET, class ShapeT, class T, class TT>
constexpr auto _diag(const tensor_base<ET, ShapeT, T> &, TT &&t);
template <class ET, class ShapeT, class T, class TT>
constexpr decltype(auto) _diag(const make_diag_result<ET, ShapeT, T> &, TT &&t);
}
template <class T>
constexpr auto diag(T &&t) -> decltype(details::_diag(t, std::forward<T>(t))) {
  return details::_diag(t, std::forward<T>(t));
}
}
