#include <gtest/gtest.h>

#include "opencv.hpp"

using namespace wheels;

TEST(third, opencv) {
  matx m1(make_shape(50, 50));

  m1 = ones(100, 100);

  auto im = imread(filesystem::path(wheels_data_dir_str) / "wheels.jpg");
  auto ime = im.eval();
  im = zeros<uint8_t>(250, 250, 3);
  im.write(filesystem::temp_directory_path() / "black.jpg");

  auto tt = tuplize(im);
  auto &tt1 = tt[0];

  uint8_t a;
}
