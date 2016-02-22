#include <gtest/gtest.h>

#include "auxmath.hpp"
#include "matrix_mul.hpp"

using namespace wheels;

TEST(math, auxmath) {
  matx A(make_shape(10, 12)), B(make_shape(10, 1));
  std::default_random_engine rng;
  randomize_fields(A, rng);
  randomize_fields(B, rng);
  matx X;
  bool b = auxmath::solve(A, B, X);
  ASSERT_TRUE(b);
  println((A * X - B).norm());
}