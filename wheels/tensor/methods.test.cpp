#include <gtest/gtest.h>

#include "methods.hpp"

using namespace wheels;

TEST(tensor, methods) {

  auto kk = ewise_mul(cube2(), cube2()).eval();
  auto kk2 = ewise_mul(ones(50, 50), zeros(50, 50)).eval();
  auto t1 = zeros(100, 100, 100);
  auto t2 = t1 + 1;
  auto &k = t2[3];
  ASSERT_TRUE(t2 == ones(100, 100, 100));

  auto a = ones(50, 30);
  auto b = a.t().t() + 1;
 
}