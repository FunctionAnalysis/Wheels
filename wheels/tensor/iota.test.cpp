#include "iota.hpp"
#include <gtest/gtest.h>

using namespace wheels;

TEST(tensor, iota) {
  size_t ns = 0;
  for (auto i : iota(50)) {
    ns += i * i;
    print(i, ' ');
  }
  auto n = iota(50).norm_squared();
  println(n);
  ASSERT_EQ(ns, n);
}