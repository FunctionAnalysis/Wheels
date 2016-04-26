#include <gtest/gtest.h>

#include "constants.hpp"
#include "ewise.hpp"
#include "matrix.hpp"
#include "tensor.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;

TEST(tensor, constants) {
  matx m = ones(4, 5);
  for (auto i : m) {
    ASSERT_TRUE(i == 1);
  }
  for (auto i : ones(5, 6, 7) * 5) {
    ASSERT_TRUE(i == 5);
  }
}