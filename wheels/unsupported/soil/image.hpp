#pragma once

#include <SOIL.h>

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
template <size_t ForceChannels = 4>
image_<uint8_t, ForceChannels> load_image(const char *file_name) {
  int width = 0, height = 0, channels = 0;
  uint8_t *data =
      SOIL_load_image(file_name, &width, &height, &channels, ForceChannels);
  image_<uint8_t, ForceChannels> im(make_shape(height, width));
  static_assert(sizeof(vec_<uint8_t, ForceChannels>) == ForceChannels, "");
  std::memcpy(im.ptr(), data, width * height * ForceChannels);
  SOIL_free_image_data(data);
  return im;
}
}
