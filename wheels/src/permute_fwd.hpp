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

template <class ET, class ShapeT, class T, size_t... Inds> class permute_result;

namespace details {
template <class ET, class ShapeT, class T, class TT, class... IndexTs>
constexpr decltype(auto) _permute(const tensor_base<ET, ShapeT, T> &, TT &&t,
                                  const IndexTs &...);
template <class ET, class ShapeT, class T, size_t... Inds, class TT,
          class... IndexTs>
constexpr decltype(auto)
_permute(const permute_result<ET, ShapeT, T, Inds...> &, TT &&t,
         const IndexTs &...);
}

template <class T, class... IndexTs>
constexpr auto permute(T &&t, const IndexTs &... inds)
    -> decltype(details::_permute(t, std::forward<T>(t), inds...)) {
  return details::_permute(t, std::forward<T>(t), inds...);
}

// transpose
namespace details {
template <class ST, class MT, class NT, class ET, class T, class TT>
constexpr auto _transpose(const tensor_base<ET, tensor_shape<ST, MT, NT>, T> &,
                          TT &&t)
    -> decltype(permute(std::forward<TT>(t), const_index<1>(), const_index<0>())) {
  return permute(std::forward<TT>(t), const_index<1>(), const_index<0>());
}
}
template <class T>
constexpr auto transpose(T &&t)
    -> decltype(details::_transpose(t, std::forward<T>(t))) {
  return details::_transpose(t, std::forward<T>(t));
}
}
