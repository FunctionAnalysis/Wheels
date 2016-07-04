#include <gtest/gtest.h>

#include "block.hpp"
#include "cartesian.hpp"
#include "constants.hpp"
#include "iota.hpp"
#include "matrix.hpp"
#include "permute.hpp"
#include "vector.hpp"

using namespace wheels;
using namespace wheels::tags;

TEST(tensor, meshgrid) {
  matx xx, yy;
  std::tie(xx, yy) = meshgrid<double>(make_shape(4, 5));
  for (int i = 0; i < 4; i++) {
    ASSERT_TRUE(xx.block(i, range(0, last)) == ones(1, 5) * i);
  }
  for (int i = 0; i < 5; i++) {
    ASSERT_TRUE(yy.block(range(0, last), i) == ones(4, 1) * i);
  }
  std::tie(xx, yy) = meshgrid<double>(make_shape(40, 40));
  ASSERT_TRUE(transpose(xx) == yy);
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