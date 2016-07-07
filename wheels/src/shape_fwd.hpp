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

#include "const_ints_fwd.hpp"

namespace wheels {

template <class T, class... SizeTs> class tensor_shape;

// is_tensor_shape
template <class T> struct is_tensor_shape;

template <class T1, class... SizeT1s, class... ShapeTs>
constexpr auto same_rank(const tensor_shape<T1, SizeT1s...> &shape1,
                         const ShapeTs &... shapes);

template <class T, class SizeT, class... SizeTs>
constexpr auto max_shape_size(const tensor_shape<T, SizeT, SizeTs...> &shape);

template <class T, class SizeT, class... SizeTs>
constexpr auto min_shape_size(const tensor_shape<T, SizeT, SizeTs...> &shape);

template <class T, class... SizeTs>
constexpr auto make_rank_sequence(const tensor_shape<T, SizeTs...> &shape);

template <class... SizeTs> constexpr auto make_shape(const SizeTs &... sizes);

template <class T, class K, class... S1s, class... S2s>
constexpr auto cat2(const tensor_shape<T, S1s...> &t1,
                    const tensor_shape<K, S2s...> &t2);

template <class T, class... Ss, class K, K... Vs>
constexpr auto cat2(const tensor_shape<T, Ss...> &a,
                    const const_ints<K, Vs...> &b);

template <class T, class... Ss, class K, K... Vs>
constexpr auto cat2(const const_ints<K, Vs...> &a,
                    const tensor_shape<T, Ss...> &b);

template <class T, class... Ss, class IntT,
          class = std::enable_if_t<std::is_integral<IntT>::value>>
constexpr auto cat2(const tensor_shape<T, Ss...> &a, const IntT &b);

template <class T, class... Ss, class IntT,
          class = std::enable_if_t<std::is_integral<IntT>::value>>
constexpr auto cat2(const IntT &a, const tensor_shape<T, Ss...> &b);

template <class ShapeOrSizeT, class T, T Times>
constexpr auto repeat_shape(const ShapeOrSizeT &s,
                            const const_ints<T, Times> &times);

template <class T, class... SizeTs, class... IndexTs>
constexpr auto permute(const tensor_shape<T, SizeTs...> &shape,
                       const IndexTs &... inds);
}
