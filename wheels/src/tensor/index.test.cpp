#include <gtest/gtest.h>

#include "constants.hpp"
#include "index.hpp"
#include "tensor.hpp"

using namespace wheels;
using namespace wheels::tags;

TEST(tensor, index) {
  vecx a({1, 2, 3, 4, 5});
  a[last] = 4;
  a[vecx_<int>({2, 3, 4})] = 1;
  ASSERT_TRUE(a == vecx({1, 2, 1, 1, 1}));
  a[where(a == 1)] = 10;
  ASSERT_TRUE(a == vecx({10, 2, 10, 10, 10}));
  println(a);
  a[last - vecxi({0, 1})] = 5;
  ASSERT_TRUE(a == vecx({10, 2, 10, 5, 5}));
  println(a);
}