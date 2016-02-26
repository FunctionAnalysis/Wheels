#include <gtest/gtest.h>

#include "../../core"
#include "../../tensor"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, iota) {
  size_t ns = 0;
  for (auto i : iota(50)) {
    ns += i * i;
    print(i, ' ');
  }
  auto n = iota(50).norm();
  println(n);
  ASSERT_EQ(sqrt(ns), n);

  // reverse
  vecx data(make_shape(100));
  std::default_random_engine rng;
  randomize_fields(data, rng);
  vecx rev_data = data[index_tags::last - iota(100)];
  for (auto i : iota(100)) {
    ASSERT_EQ(data[i], rev_data[index_tags::last - i]);
  }
}

TEST(tensor, range) {
  ASSERT_TRUE(range(0, 2, 5) == vecxi(0, 2, 4));
  ASSERT_TRUE(range(0, 2, 6) == vecxi(0, 2, 4));
  ASSERT_TRUE(range(0, 2, 7) == vecxi(0, 2, 4, 6));
  ASSERT_TRUE(range(0, 2, 1) == vecxi(0));
  println(range(0, 2, 0));
  ASSERT_TRUE(range(0, 2, 0) == vecxi());
  println(range(0, 2, -1).shape());
  ASSERT_TRUE(range(0, 2, -1) == vecxi());
  ASSERT_TRUE(range(5, 2, -1) == vecxi());
  ASSERT_TRUE(range(5, -1, -1) == vecxi(5, 4, 3, 2, 1, 0));
  ASSERT_TRUE(range(5, -2, -1) == vecxi(5, 3, 1));
  println(range(5, -2, 0));
  ASSERT_TRUE(range(5, -2, 0) == vecxi(5, 3, 1));
  ASSERT_TRUE(range(5, -3, 0) == vecxi(5, 2));
  ASSERT_TRUE(range(5, -4, 1) == vecxi(5));
  ASSERT_TRUE(range(5, -5, 1) == vecxi(5));

  ASSERT_TRUE(range(0_symbol, 2, 1_symbol)(0, 5) == range(0, 2, 5));
}