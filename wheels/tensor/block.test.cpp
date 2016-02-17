#include <gtest/gtest.h>

#include "constants.hpp"
#include "block.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, block) {
  ASSERT_TRUE(blockwise(ones(5ull, 4_sizec, 3_sizec), 1_c).sum() ==
              constants(make_shape(4_c, 3_c), 5));
  ASSERT_TRUE(blockwise(ones(5ull, 4_sizec, 3_sizec), 2_c).sum() ==
              constants(make_shape(3_c), 20));
}

struct A {
    A(){}
    A(A &&) = default;
    //A(const A &) = default;
    A & operator=(A &&) = default;
};

TEST(test, test) {
    A a;
    A aa;
    aa = std::move(a);
}