#include <gtest/gtest.h>

#include "cat.hpp"
#include "ewise.hpp"
#include "tensor_map.hpp"
#include "shape.hpp"
#include "tensor.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, cat) {
  auto a = "12345"_ts;
  auto b = "abcdefg"_ts;
  auto ab = cat(a, b, "12345"_ts);

  ASSERT_FALSE(ab == "12345abcdefg123456"_ts);
  ASSERT_TRUE(ab != "12345abcdefg123456"_ts);

  ASSERT_FALSE(ab != "12345abcdefg12345"_ts);
  ASSERT_TRUE(ab == "12345abcdefg12345"_ts);

  auto result =
      cat(vec3(1, 2, 3), vec2(4, 5), vecx({6, 7, 8, 9, 10}), vecx({11}))
          .ewised() == vecx({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  result.for_each([](bool b) { ASSERT_TRUE(b); });

  for_each_element(behavior_flag<unordered>(), [](bool b) { ASSERT_TRUE(b); },
                   cat(vec3(1, 2, 3), vec2(4, 5), vecx({6})).ewised() ==
                       vecx({1, 2, 3, 4, 5, 6}));
}