#include <gtest/gtest.h>

#include "../tensor.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, permute) {
  auto t = zeros(1, 2, 3, 4, 5).eval();
  std::default_random_engine rng;
  randomize_fields(t, rng);
  ASSERT_TRUE(t != zeros(t.shape()));
  auto permuted = permute(t, 2_c, 4_c, 0_c, 3_c, 1_c).eval();
  for_each_subscript(
      t.shape(), [&t, &permuted](auto s0, auto s1, auto s2, auto s3, auto s4) {
        ASSERT_EQ(t(s0, s1, s2, s3, s4), permuted(s2, s4, s0, s3, s1));
      });
}