#include <gtest/gtest.h>

#include "../tensor.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, reformulate) {
  std::default_random_engine rng;
  auto a = rand(make_shape(10, 20, 30), rng);
  auto b1 = reformulate(a, make_shape(30, 20, 10), [](auto i, auto j, auto k) {
    return std::make_tuple(k, j, i);
  });
  auto b2 = permute(a, 2_c, 1_c, 0_c);
  ASSERT_TRUE(b1 == b2);

  auto c1 = reformulate(a, make_shape(60, 30), [](auto i, auto j) {
    return std::make_tuple(0, i % 20, j);
  });
  auto c2 =
      cat(subtensor_at(a, 0), subtensor_at(a, 0), subtensor_at(a, 0)).eval();

  ASSERT_TRUE(c1 == c2);
}

TEST(tensor, DISABLED_reformulate2) {
  std::default_random_engine rng;
  auto a = rand(make_shape(10, 20, 30), rng);
  auto c1 = reformulate(a, make_shape(60, 30), [](auto i, auto j) {
    return std::make_tuple(0, i % 20, j);
  });

  auto c3 = repeat(subtensor_at(a, 0), 3, 1);
  ASSERT_TRUE(c1 == c3);
}