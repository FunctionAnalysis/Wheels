#include <gtest/gtest.h>

#include "../../tensor"

using namespace wheels;

TEST(tensor, meshgrid) {
  matx xx, yy;
  std::tie(xx, yy) = meshgrid<double>(make_shape(4, 5));
  println(xx);
  println(yy);
  std::tie(xx, yy) = meshgrid<double>(make_shape(40, 40));
  ASSERT_TRUE(xx.t() == yy);
}

TEST(tensor, coordinate) {
  auto s = iota(make_shape(4, 5));
  size_t i = 0;
  for (auto &&c : coordinate(make_shape(4, 5))) {
    ASSERT_TRUE(s(c[0], c[1]) == i++);
  }
}

TEST(tensor, cart_prod) {
  auto s = iota(make_shape(4, 5, 6));
  size_t i = 0;
  for (auto &&c : cart_prod(iota<int>(4), iota<size_t>(5), iota<short>(6))) {
    ASSERT_TRUE(s(std::get<0>(c), std::get<1>(c), std::get<2>(c)) == i++);
  }
}