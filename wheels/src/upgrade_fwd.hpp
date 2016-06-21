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
#include "shape_fwd.hpp"

namespace wheels {

template <class ET, class ShapeT, class InputT, class ExtShapeT, class ExtFunT>
class upgrade_result;

namespace details {
template <class ET, class InputShapeT, class InputET, class InputT,
          class InputTT, class ExtShapeT, class ExtFunT, size_t... ExtIs>
constexpr auto _upgrade_by(const tensor_base<InputET, InputShapeT, InputT> &,
                           InputTT &&input, const ExtShapeT &extshape,
                           ExtFunT extfun,
                           const const_ints<size_t, ExtIs...> &);
}

// upgrade_by
template <class ET, class InputT, class ST, class... SizeTs, class ExtFunT>
constexpr auto upgrade_by(InputT &&input, const tensor_shape<ST, SizeTs...> &es,
                          ExtFunT ef)
    -> decltype(details::_upgrade_by<ET>(input, std::forward<InputT>(input), es,
                                         ef, make_rank_sequence(es))) {
  return details::_upgrade_by<ET>(input, std::forward<InputT>(input), es, ef,
                                  make_rank_sequence(es));
}

// upgrade_as_repeated
namespace details {
template <class ET, class ShapeT, class T, class InputT, class ST,
          class... SizeTs>
constexpr auto _upgrade_as_repeated(const tensor_base<ET, ShapeT, T> &,
                                    InputT &&input,
                                    const tensor_shape<ST, SizeTs...> &es);
}

template <class InputT, class ST, class... SizeTs>
constexpr auto upgrade_as_repeated(InputT &&input,
                                   const tensor_shape<ST, SizeTs...> &es) {
  return details::_upgrade_as_repeated(input, std::forward<InputT>(input), es);
}

// upgrade_all
template <class InputT>
constexpr decltype(auto) upgrade_all(InputT &&input);
}