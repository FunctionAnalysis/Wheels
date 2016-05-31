#include <gtest/gtest.h>

#include "remap.hpp"
#include "tensor.hpp"
#include "cartesian.hpp"
#include "ewise.hpp"
#include "constants.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, resample) {
  auto shape = make_shape((size_t)200, (size_t)300);
  auto xx = meshgrid(shape, 0_c);
  auto yy = meshgrid(shape, 1_c);
  auto zz = (sin((xx - 10.0) / 5.0).ewised() * cos((yy - 15.0) / 5.0)).eval();
  auto zzre = resample(zz, make_shape((size_t)100, (size_t)150)).eval();
  auto zzre2 = resample(zz, make_shape((size_t)400, (size_t)600)).eval();
  auto zz2 = zz.ewised().transform([](double e) {return vec3(sin(e), cos(e), e);}).eval();
  auto zz2re = resample(zz2, make_shape((size_t)100, (size_t)150)).eval();
  auto zz2re2 = resample(zz2, make_shape((size_t)400, (size_t)600)).eval();
  println(zz2re2.shape());
}