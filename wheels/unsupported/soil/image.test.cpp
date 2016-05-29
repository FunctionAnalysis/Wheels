#include <gtest/gtest.h>

#include "image.hpp"

using namespace wheels;

TEST(soil, image) {
  image3f32 im(make_shape(500, 500));

  println(im.shape());
}