#include <gtest/gtest.h>
#include "map.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, map) {
  auto s1 = "1234567"_ts;
  auto s2 = "abcdefghijk"_ts;
 
  element_at_index(s1, 5);
}