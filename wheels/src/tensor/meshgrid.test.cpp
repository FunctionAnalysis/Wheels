#include <gtest/gtest.h>

#include "meshgrid.hpp"
#include "permute.hpp"
#include "tensor.hpp"

using namespace wheels;

TEST(tensor, meshgrid) {
  matx xx, yy;
  std::tie(xx, yy) = meshgrid<double>(make_shape(4, 5));
  println(xx);
  println(yy);
  std::tie(xx, yy) = meshgrid<double>(make_shape(40, 40));
  ASSERT_TRUE(xx.t() == yy);
}