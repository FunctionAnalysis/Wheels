#include "image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

namespace wheels {
namespace details {
template <size_t ForceChannels>
image_<uint8_t, ForceChannels> _load_image(const char *file_name) {
  int width = 0, height = 0, channels = 0;
  uint8_t *data =
      stbi_load(file_name, &width, &height, &channels, ForceChannels);
  image_<uint8_t, ForceChannels> im(make_shape(height, width));
  static_assert(sizeof(vec_<uint8_t, ForceChannels>) == ForceChannels, "");
  std::memcpy(im.ptr(), data, width * height * ForceChannels);
  stbi_image_free(data);
  return im;
}
}

// load_image
image4u8 load_image_rgba(const char *file_name) {
  return details::_load_image<4>(file_name);
}
image3u8 load_image_rgb(const char *file_name) {
  return details::_load_image<3>(file_name);
}
image1u8 load_image_gray(const char *file_name) {
  return details::_load_image<1>(file_name);
}
}