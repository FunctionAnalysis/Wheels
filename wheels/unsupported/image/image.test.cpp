#include <gtest/gtest.h>

#include "../../src/diagonal.hpp"
#include "../../src/upgrade.hpp"
#include "../../src/iota.hpp"
#include "../../src/permute.hpp"
#include "../../src/matrix.hpp"
#include "../../src/remap.hpp"

#include "image.hpp"

using namespace wheels;
using namespace wheels::tags;

TEST(image, load) {
  auto im = load_image_rgb(wheels_data_dir_str "/wheels_color.jpg");
  auto im_norm = im.ewised()
                     .transform([](auto &&rgb) -> double {
                       return rgb.ewised().cast<by_static, double>().norm();
                     })
                     .eval();
  auto im_transposed = im.t().eval();
  auto imd = im.ewised()
                 .transform([](auto &e) {
                   return e.ewised().cast<by_static, double>();
                 })
                 .eval();
	auto imd3 = im.ewised().cast<by_construct, vec3>().eval();
  auto imd2 = imd;
  imd2.block(range(0, 50, last), range(0, last)) +=
      (vec3(255, 255, 255)).eval().scalarized();
  auto imdiff = (imd2 - imd).eval();
  auto im_scaled = resample(imd, make_shape((size_t)600, (size_t)600)).eval();
  println(im_scaled.shape());
}