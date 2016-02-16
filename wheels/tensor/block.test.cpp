#include <gtest/gtest.h>

#include "constants.hpp"
#include "block.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, block) {

  auto t = ones(5ull, 4_sizec, 3_sizec);
  auto tb = block_at(t, 3).eval();

  auto tbb = blockwise(t, 2_c);
  ASSERT_TRUE(tbb.sum() == constants(make_shape(3_c), 20));
}