#include <gtest/gtest.h>

#include "image.hpp"

using namespace wheels;

TEST(image, image) {
  auto im = load_image_rgb(wheels_data_dir_str"/wheels.jpg");
  println(im.shape());
}