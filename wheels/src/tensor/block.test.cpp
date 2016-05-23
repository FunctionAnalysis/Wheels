#include <gtest/gtest.h>

#include "aligned.hpp"
#include "block.hpp"
#include "constants.hpp"
#include "iota.hpp"
#include "reshape.hpp"
#include "tensor.hpp"

using namespace wheels;
using namespace wheels::tags;
using namespace wheels::literals;

TEST(tensor, block) {
  // 1, 2, 3, 4, 5,
  // 6, 7, 8, 9, 10,
  // 11, 12, 13, 14, 15,
  // 16, 17, 18, 19, 20
  matx a(make_shape(4, 5), {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                            16, 17, 18, 19, 20});
  ASSERT_TRUE(a.vectorized() == iota(20) + 1);

  auto r = range(0, 5);
  auto s = r.shape();
  ASSERT_TRUE(s == make_shape(6_c));

  println(a.size(const_index<1>()));
  println(range(0, 5));
  println(a.block(0, range(0, last)).shape());
  ASSERT_TRUE(a.block(0, range(0, last)) == rowvecx({1.0, 2.0, 3.0, 4.0, 5.0}));
  ASSERT_TRUE(a.block(1, range(0, last)) == rowvecx({6, 7, 8, 9, 10}));

  ASSERT_TRUE(a.block(0, range(0, 2, last)) == rowvecx({1, 3, 5}));
  ASSERT_TRUE(a.block(1, range(0, 2, last)) == rowvecx({6, 8, 10}));

  // 1, 3, 5
  // 16, 18, 20
  println(a.block(range(0, 3, last), range(0, 2, last)));
  println(matx(make_shape(2, 3), with_elements, 1, 3, 5, 16, 18, 20));
  ASSERT_TRUE(a.block(range(0, 3, last), range(0, 2, last)) ==
              matx(make_shape(2, 3), with_elements, 1, 3, 5, 16, 18, 20));

  // 5, 3, 1
  // 20, 18, 16
  println(a.block(range(0, 3, last), range(last, -2, 0)));
  println(matx(make_shape(2, 3), with_elements, 5, 3, 1, 20, 18, 16));
  ASSERT_TRUE(a.block(range(0, 3, last), range(last, -2, 0)) ==
              matx(make_shape(2, 3), with_elements, 5, 3, 1, 20, 18, 16));
}