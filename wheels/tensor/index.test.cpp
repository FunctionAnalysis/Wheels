#include <gtest/gtest.h>

#include "constants.hpp"
#include "index.hpp"
#include "tensor.hpp"

using namespace wheels;

TEST(tensor, index) {
  vecx a(1, 2, 3, 4, 5);
  a[index_tags::last] = 4;
  a[vecx_<int>(2, 3, 4)] = 1;
  ASSERT_TRUE(a == vecx(1, 2, 1, 1, 1));
  a[where(a == 1)] = 10;
  ASSERT_TRUE(a == vecx(10, 2, 10, 10, 10));
  println(a);
}