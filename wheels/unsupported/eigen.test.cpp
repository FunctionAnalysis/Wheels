#include <gtest/gtest.h>
#include "eigen.hpp"

using namespace wheels;
using namespace Eigen;

TEST(unsupported, eigen) {
  using m_t = Matrix<double, Dynamic, Dynamic, ColMajor | DontAlign, 5, 6>;
  m_t m = m_t::Zero(3, 4);
  for (int i = 0; i < m.size(); i++) {
    m(i) = i + 1;
  }
  std::cout << m << '\n';
  auto d = m.data();

  std::vector<int> v;
}