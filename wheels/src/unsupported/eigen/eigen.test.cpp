#include <gtest/gtest.h>

#include "../../tensor/matrix.hpp"
#include "../../tensor/vector.hpp"

#include "eigen.hpp"

using namespace wheels;
using namespace Eigen;

TEST(unsupported, eigen) {
  Eigen::VectorXd ev1 = Eigen::VectorXd::Random(100);
  auto v1 = map(ev1);
  for (int i = 0; i < v1.numel(); i++) {
    ASSERT_EQ(v1[i], ev1[i]);
  }
  Eigen::MatrixXd em1 = Eigen::MatrixXd::Random(20, 30);
  auto m1 = map(em1);
  ASSERT_EQ(em1.rows(), m1.rows());
  ASSERT_EQ(em1.cols(), m1.cols());
  for (int i = 0; i < m1.rows(); i++) {
    for (int j = 0; j < m1.cols(); j++) {
      ASSERT_EQ(m1(i, j), em1(i, j));
    }
  }
}