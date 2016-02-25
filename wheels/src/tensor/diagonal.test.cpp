#include <gtest/gtest.h>

#include "diagonal.hpp"

using namespace wheels;

TEST(tensor, diagonal) {
  auto I = eye(5000);
  auto n = I.norm();
  ASSERT_TRUE(n == sqrt(5000));
  ASSERT_TRUE(I.sum() == 5000);
  decltype(auto) Icore = diag(I);
  ASSERT_TRUE(Icore == ones(make_shape(5000)));
}