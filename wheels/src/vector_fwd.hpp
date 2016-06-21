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

template <class ET1, class ShapeT1, class T1, class ET2, class ShapeT2,
          class T2>
constexpr auto distance(const tensor_base<ET1, ShapeT1, T1> &t1,
                        const tensor_base<ET2, ShapeT2, T2> &t2);

template <class ET1, class ShapeT1, class T1, class ET2, class ShapeT2,
          class T2>
auto dot(const tensor_base<ET1, ShapeT1, T1> &t1,
         const tensor_base<ET2, ShapeT2, T2> &t2);

template <class E1, class ST1, class NT1, class T1, class E2, class ST2,
          class NT2, class T2>
constexpr auto cross(const tensor_base<E1, tensor_shape<ST1, NT1>, T1> &a,
                     const tensor_base<E2, tensor_shape<ST2, NT2>, T2> &b);
}