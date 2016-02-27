#include <gtest/gtest.h>

#include "../../core"
#include "../../tensor"

#include "auxmath.hpp"
#include "matrix_mul.hpp"

using namespace wheels;

TEST(math, solve) {
  std::default_random_engine rng;
  for (size_t i : iota(10)) {
    for (size_t j : iota(10)) {
      for (size_t k : iota(10)) {
        matx A(make_shape(i + 1, i + 1 + j)), B(make_shape(i + 1, k + 1));
        randomize_fields(A, rng);
        randomize_fields(B, rng);
        bool b = false;
        println(type_of(auxmath::solve(A, B, &b)));
        ASSERT_TRUE((A * auxmath::solve(A, B, &b) - B).norm() < 1e-3);
        ASSERT_TRUE(b);
      }
      {
        matx A(make_shape(i + 1, i + 1 + j));
        vecx B(make_shape(i + 1));
        randomize_fields(A, rng);
        randomize_fields(B, rng);
        bool b = false;
        ASSERT_TRUE((A * auxmath::solve(A, B, &b) - B).norm() < 1e-3);
        ASSERT_TRUE(b);
      }
    }
  }
}

TEST(DISABLED_math, inverse) {
  std::default_random_engine rng;
  for (size_t i : iota(10) + 2) {
    matx A(make_shape(i, i));
    randomize_fields(A, rng);
    bool b = false;
    auto X = auxmath::inverse(A, &b);
    ASSERT_TRUE((A * X - eye(i)).norm() < 1e-3);
    ASSERT_TRUE(b);
  }
}