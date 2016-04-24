#include <gtest/gtest.h>

#include "../../core"
#include "../../tensor"

#include "auxmath.hpp"

using namespace wheels;

TEST(math, solve) {
  std::default_random_engine rng;
  for (size_t i : iota(10)) {
    for (size_t j : iota(10)) {
      for (size_t k : iota(10)) {
        auto A = rand(make_shape(i + 1, i + 1 + j), rng);
        auto B = rand(make_shape(i + 1, k + 1), rng);
        bool b = false;
        ASSERT_TRUE((A * auxmath::solve(A, B, &b) - B).norm() < 1e-3);
        ASSERT_TRUE(b);
      }
      {
        auto A = rand(make_shape(i + 1, i + 1 + j), rng);
        auto B = rand(make_shape(i + 1), rng);
        bool b = false;
        ASSERT_TRUE((A * auxmath::solve(A, B, &b) - B).norm() < 1e-3);
        ASSERT_TRUE(b);
      }
    }
  }
}

TEST(math, inverse) {
  std::default_random_engine rng;
  for (auto i : iota(100) + 2) {
    auto A = rand(make_shape(i, i), rng);
    bool b = false;
    auto X = auxmath::inverse(A, &b);
    ASSERT_TRUE((A * X - eye(i)).norm() < 1e-3);
    ASSERT_TRUE(b);
  }
}