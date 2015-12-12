#include <gtest/gtest.h>

#include "methods.hpp"

using namespace wheels;

TEST(tensor, methods) {
  auto kk = ewise_mul(cube2(), cube2()).eval();
  auto kk2 = ewise_mul(ones(50, 50), zeros(50, 50)).eval();
  auto t1 = zeros(100, 100, 100);
  auto t2 = t1 + 1;
  auto &k = t2[3];
  ASSERT_TRUE(t2 == ones(100, 100, 100));

  auto a = ones(50, 30);
  auto b = a.t().t() + 1;
}

TEST(tensor, methods2) {
  mat_<double, 2, 3> t1 = ones(2, 3);
  auto r1 = sin(t1);
  auto r2 = r1 + t1 * 2.0;
  auto rr = min(t1, r2);

  for_each([](double e) { ASSERT_EQ(e, min(1.0, sin(1) + 2)); }, rr);
  for_each([](double e) { ASSERT_EQ(e, min(1.0, sin(1) + 2)); }, rr.eval());
  for_each([](double e) { ASSERT_EQ(e, min(1.0, sin(1) + 2)); }, rr.t());
  for_each([](double e) { ASSERT_EQ(e, min(1.0, sin(1) + 2)); }, rr.t().t());
}