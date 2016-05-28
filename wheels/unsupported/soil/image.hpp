#pragma once

#include <SOIL.h>

#include "../../src/tensor.hpp"

namespace wheels {
// load_image
template <size_t ForceChannels = 4>
image_<vec_<uint8_t, ForceChannels>> load_image(const char *file_name) {
  int width = 0, height = 0, channels = 0;
  uint8_t *data =
      SOIL_load_image(file_name, &width, &height, &channels, ForceChannels);
  image_<vec_<uint8_t, ForceChannels>> im(make_shape(height, width));
  static_assert(sizeof(vec_<uint8_t, ForceChannels>) == ForceChannels, "");
  std::memcpy(im.ptr(), data, width * height * ForceChannels);
  SOIL_free_image_data(data);
  return im;
}
}
