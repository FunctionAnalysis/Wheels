#include <gtest/gtest.h>

#include "diagonal.hpp"
#include "ewise.hpp"
#include "iota.hpp"
#include "map.hpp"
#include "tensor.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, map) {
  ASSERT_TRUE("1234567"_ts == range('1', '7'));
  ASSERT_TRUE(""_ts.numel() == 0);
  ASSERT_TRUE("1234567"_ts == "abcdefg"_ts - 'a' + '1');

  int data[2][2] = {{1, 0}, {0, 1}};
  ASSERT_TRUE(map(data) == eye<int>(2, 2));
}