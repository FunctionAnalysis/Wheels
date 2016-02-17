#include <gtest/gtest.h>

#include "constants.hpp"
#include "index.hpp"
#include "tensor.hpp"

using namespace wheels;

TEST(tensor, index) {
  vecx a(1, 2, 3, 4, 5);
  a[vecx_<int>(2, 3, 4)] = ones(3);
  ASSERT_TRUE(a == vecx(1, 2, 1, 1, 1));
}