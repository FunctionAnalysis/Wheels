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

// subtensor_view
template <class ET, class SubShapeT, class InputT, size_t FixedRank>
class subtensor_view;

namespace detail {
template <class ET, class ShapeT, class T, class TT, class... SubTs>
constexpr auto _subtensor_at(const tensor_base<ET, ShapeT, T> &, TT &&input,
                             const SubTs &... subs);
}

template <class T, class... SubTs>
constexpr auto subtensor_at(T &&input, const SubTs &... subs)
    -> decltype(detail::_subtensor_at(input, std::forward<T>(input),
                                       subs...)) {
  return detail::_subtensor_at(input, std::forward<T>(input), subs...);
}

// downgrade_view
template <class ET, class ShapeT, class InputT, size_t FixedRank>
class downgrade_view;

template <class ET, class ShapeT, class InputT, class ExtShapeT, class ExtFunT>
class upgrade_result;
namespace upgrade_result_ops {
struct as_subtensor;
}

namespace detail {
template <class ET, class ShapeT, class InputT, class TT, size_t FixedRank>
constexpr auto _downgrade(const tensor_base<ET, ShapeT, InputT> &, TT &&input,
                          const const_size<FixedRank> &);
// downgrade an upgraded tensor
template <class ET, class ShapeT, class InputT, class ExtShapeT, class TT>
constexpr decltype(auto)
_downgrade(const upgrade_result<ET, ShapeT, InputT, ExtShapeT,
                                upgrade_result_ops::as_subtensor> &,
           TT &&input, const const_size<ExtShapeT::rank> &);
}
template <class InputT, class K, K FixedRank>
constexpr auto downgrade(InputT &&input, const const_ints<K, FixedRank> &r)
    -> decltype(detail::_downgrade(input, std::forward<InputT>(input),
                                    const_size<(size_t)FixedRank>())) {
  return detail::_downgrade(input, std::forward<InputT>(input),
                             const_size<(size_t)FixedRank>());
}
}
