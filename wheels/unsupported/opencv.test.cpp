#include <gtest/gtest.h>

#include "../core/time.hpp"
#include "../tensor.hpp"
#include "opencv.hpp"

using namespace wheels;

TEST(third, opencv) {
  matx m1(make_shape(50, 50));

  m1 = ones(100, 100);

  matx_<vec3> mmm(make_shape(50, 50));

  auto im = imread(filesystem::path(wheels_data_dir_str) / "wheels.jpg");

  auto nim = remap(extend_as_subtensor(im), make_shape(800, 200, 3),
                   [](auto y, auto x, auto c) {
                     return vec3(x / 2.0 + 300.0, y / 3.0, c / 2.0);
                   })
                 .eval();
  println(type_of(nim));
}
