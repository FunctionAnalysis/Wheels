#include <gtest/gtest.h>

#include "map.hpp"
#include "tensor.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, map) {
  tstring s1 = "1234567"_ts;
  tstring s2 = "abcdefghijk"_ts;
 
  element_at_index(s1, 5);

  int a1[3][2][2] = {123};
  auto ta1 = map(a1).eval();
}