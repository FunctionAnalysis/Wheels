#include <gtest/gtest.h>

#include <random>

#include "constants.hpp"
#include "permute.hpp"
#include "tensor.hpp"
#include "matrix.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, permute) {
  std::default_random_engine rng;
  auto t = rand(make_shape(1, 2, 3, 4, 5), rng);
  ASSERT_TRUE(t != zeros(t.shape()));
  auto permuted = t.permuted(2_c, 4_c, 0_c, 3_c, 1_c);
  for_each_subscript(
      t.shape(), [&t, &permuted](auto s0, auto s1, auto s2, auto s3, auto s4) {
        ASSERT_EQ(t(s0, s1, s2, s3, s4), permuted(s2, s4, s0, s3, s1));
      });
  decltype(auto) permuted2 =
      permute(permute(t, 4_c, 3_c, 2_c, 1_c, 0_c), 4_c, 3_c, 2_c, 1_c, 0_c);
  ASSERT_TRUE(&t == &permuted2);
}

TEST(tensor, transpose) {
  std::default_random_engine rng;
  auto m = rand(make_shape(3_c, 4), rng);

  ASSERT_TRUE(m.t() == m.t());
  ASSERT_TRUE(m.t().t() == m);
  ASSERT_TRUE(type_of(m.t().t()) == type_of(m));
}