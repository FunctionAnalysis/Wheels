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

#include "image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#undef STB_IMAGE_IMPLEMENTATION

namespace wheels {
namespace detail {
template <size_t ForceChannels>
image_<uint8_t, ForceChannels> _load_image(const char *file_name) {
  int width = 0, height = 0, channels = 0;
  uint8_t *data =
      stbi_load(file_name, &width, &height, &channels, ForceChannels);
  image_<uint8_t, ForceChannels> im(make_shape(height, width));
  static_assert(sizeof(vec_<uint8_t, ForceChannels>) == ForceChannels, "");
  memcpy(im.ptr(), data, width * height * ForceChannels);
  stbi_image_free(data);
  return im;
}
}

// load_image
image4u8 load_image_rgba(const char *file_name) {
  return detail::_load_image<4>(file_name);
}
image3u8 load_image_rgb(const char *file_name) {
  return detail::_load_image<3>(file_name);
}
image1u8 load_image_gray(const char *file_name) {
  return detail::_load_image<1>(file_name);
}
}