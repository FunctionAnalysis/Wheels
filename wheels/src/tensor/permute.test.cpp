#include <gtest/gtest.h>

#include "../../core"
#include "../../tensor"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, permute) {
  auto t = zeros(1, 2, 3, 4, 5).eval();
  std::default_random_engine rng;
  randomize_fields(t, rng);
  ASSERT_TRUE(t != zeros(t.shape()));
  auto permuted = permute(t, 2_c, 4_c, 0_c, 3_c, 1_c);
  println(type_of(permuted));
  for_each_subscript(
      t.shape(), [&t, &permuted](auto s0, auto s1, auto s2, auto s3, auto s4) {
        ASSERT_EQ(t(s0, s1, s2, s3, s4), permuted(s2, s4, s0, s3, s1));
      });
  auto permuted2 = permute(std::move(permuted), 4_c, 2_c, 0_c, 1_c, 3_c);
  println(type_of(permuted2));
}

TEST(tensor, transpose) {
  auto m = zeros(make_shape(3_c, 4)).eval();
  std::default_random_engine rng;
  randomize_fields(m, rng);

  ASSERT_TRUE(m.t() == m.t());
  ASSERT_TRUE(m.t().t() == m);
  ASSERT_TRUE(type_of(m.t().t()) == type_of(m));
  println(type_of(m.t()));
  println(type_of(m.t().t()));
}