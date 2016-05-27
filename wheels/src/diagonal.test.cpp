#include <gtest/gtest.h>

#include "diagonal.hpp"
#include "tensor.hpp"

using namespace wheels;

TEST(tensor, diagonal) {
  ASSERT_TRUE(eye(5000).norm() == sqrt(5000));
  ASSERT_TRUE(eye(5000).sum() == 5000);
  ASSERT_TRUE(diag(eye(5000)) == ones(make_shape(5000)));
  ASSERT_TRUE(make_diag(vecx({2.0, 3.0, 4.0})) == matx(make_shape(3, 3),
                                                       with_elements, 2.0, 0.0,
                                                       0, 0, 3, 0, 0, 0, 4));
}