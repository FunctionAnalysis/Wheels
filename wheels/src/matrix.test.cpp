#include <gtest/gtest.h>

#include "diagonal.hpp"
#include "matrix.hpp"
#include "reshape.hpp"
#include "tensor.hpp"

using namespace wheels;

TEST(matrix, row_col) {
  ASSERT_TRUE(eye(4).row(0).vectorized() == vec4(1, 0, 0, 0));
  ASSERT_TRUE(eye(4).row(1).vectorized() == vec4(0, 1, 0, 0));
  ASSERT_TRUE(eye(4).row(2).vectorized() == vec4(0, 0, 1, 0));
  ASSERT_TRUE(eye(4).row(3).vectorized() == vec4(0, 0, 0, 1));

  ASSERT_TRUE(eye(4).col(0).vectorized() == vec4(1, 0, 0, 0));
  ASSERT_TRUE(eye(4).col(1).vectorized() == vec4(0, 1, 0, 0));
  ASSERT_TRUE(eye(4).col(2).vectorized() == vec4(0, 0, 1, 0));
  ASSERT_TRUE(eye(4).col(3).vectorized() == vec4(0, 0, 0, 1));

  std::default_random_engine rng;
  for (int i = 0; i < 100; i++) {
    auto m = rand(make_shape(20, 30), rng);
    for (int r = 0; r < m.rows(); r++) {
      for (int c = 0; c < m.cols(); c++) {
        ASSERT_EQ(m.row(r)[c], m(r, c));
        ASSERT_EQ(m.col(c)[r], m(r, c));
      }
    }
  }
}

TEST(matrix, matrix_ops) {
	std::default_random_engine rng;
	auto identity = eye(make_shape(4ull, 4ull)).eval();
	auto rand_mat = rand(make_shape(4ull, 4ull), rng);
  println(translate(eye(4), vec3(1, 1, 1)));
  println(rotate(eye(4), M_PI_2, vec3(1, 0, 0)));
  println(scale(eye(4), vec3(0.1, 0.2, 0.3)));
  auto m = translate(translate(rotate(rotate(eye(4), M_PI_2, vec3(1, 0, 0)),
                                      -M_PI_2, vec3(1, 0, 0)),
                               vec3(1, 2, 3)),
                     vec3(-1, -2, -3));
  ASSERT_LE((m - eye(4)).norm(), 1e-3);
}