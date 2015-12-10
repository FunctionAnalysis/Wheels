#include <gtest/gtest.h>

#include "methods.hpp"

using namespace wheels;

TEST(tensor, methods) {

  auto t1 = zeros(100, 100, 100);
  auto t2 = t1 + 1;
  auto & k = t2[3];
  ASSERT_TRUE(t2 == ones(100, 100, 100));
}