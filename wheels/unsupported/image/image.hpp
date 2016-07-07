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

#include "../../src/tensor.hpp"

namespace wheels {
// image_
template <class T, size_t C> using image_ = matx_<vec_<T, C>>;
using image1b = image_<bool, 1>;
using image3b = image_<bool, 3>;
using image4b = image_<bool, 4>;

using image1u8 = image_<uint8_t, 1>;
using image3u8 = image_<uint8_t, 3>;
using image4u8 = image_<uint8_t, 4>;

using image1i32 = image_<int32_t, 1>;
using image3i32 = image_<int32_t, 3>;
using image4i32 = image_<int32_t, 4>;

using image1f32 = image_<float, 1>;
using image3f32 = image_<float, 3>;
using image4f32 = image_<float, 4>;

using image1f64 = image_<double, 1>;
using image3f64 = image_<double, 3>;
using image4f64 = image_<double, 4>;

// load_image
image4u8 load_image_rgba(const char *file_name);
image3u8 load_image_rgb(const char * file_name);
image1u8 load_image_gray(const char * file_name);
}
