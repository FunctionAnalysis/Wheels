#include <gtest/gtest.h>

#include "../../src/diagonal.hpp"
#include "../../src/upgrade.hpp"
#include "../../src/iota.hpp"
#include "../../src/permute.hpp"
#include "../../src/matrix.hpp"

#include "image.hpp"

using namespace wheels;

TEST(image, load) {
  auto im = load_image_rgb(wheels_data_dir_str "/wheels_color.jpg");
  auto im_norm = im.ewised()
                 .transform([](auto &&rgb) -> double {
                   return rgb.ewised().cast<double>().norm();
                 })
                 .eval();
  auto im_transposed = im.t().eval();
  auto im_minused = (im - vec_<uint8_t, 3>(100, 100, 100)).eval();
  auto im_scaled = (im * 0.5).eval();
}