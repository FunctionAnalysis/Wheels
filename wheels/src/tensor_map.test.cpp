#include <gtest/gtest.h>

#include "diagonal.hpp"
#include "ewise.hpp"
#include "iota.hpp"
#include "tensor_map.hpp"
#include "tensor.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, map) {
  auto s1 = "412342513451345134f151f143511351"_ts;
  auto s2 = L"412342513451345134f151f143511351"_ts;
  auto s3 = u"412342513451345134f151f143511351"_ts;
  auto s4 = U"412342513451345134f151f143511351"_ts;

  ASSERT_TRUE("1234567"_ts == range('1', '7'));
  ASSERT_TRUE(""_ts.numel() == 0);
  ASSERT_TRUE("1234567"_ts == "abcdefg"_ts - 'a' + '1');

  int data[2][2] = {{1, 0}, {0, 1}};
  ASSERT_TRUE(map(data) == eye<int>(2, 2));
}