#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>

#include "../core/serialize.hpp"
#include "../core/types.hpp"

#include "shape.hpp"

using namespace wheels;

TEST(tensor, shape) {

  using namespace wheels::literals;

  auto s1 = make_shape(1_c, 2_c, 4, 5);
  std::cout << s1 << std::endl;
  auto test = s1[0_c] == 1_c && s1.at(1_c) == 2_c;
  static_assert(test, "");

  ASSERT_TRUE(s1.at(2_c) == 4);
  ASSERT_TRUE(s1.magnitude() == 40);

  ASSERT_TRUE(max_shape_size(s1) == 5);

  s1.resize(2_c, 5);
  ASSERT_TRUE(s1[2_c] == 5);

  ASSERT_TRUE(s1.magnitude() == 50);

  auto s2 = make_shape(1_c, 2_c, 5_c, 5_c);
  std::cout << s2 << std::endl;
  auto ns2 = make_shape(8_c);

  std::cout << ns2 << std::endl;

  ASSERT_TRUE(s1 == s2);
  ASSERT_TRUE(max_shape_size(s2) == 5);

  std::vector<size_t> inds;
  for_each_subscript(
      s2, [&s2, &inds](auto... subs) { inds.push_back(sub2ind(s2, subs...)); });
  std::vector<size_t> inds2(inds.size());
  std::iota(inds2.begin(), inds2.end(), 0);

  ASSERT_TRUE(inds == inds2);

  constexpr auto s3 = make_shape(1_sizec, 5_c);
  auto m3 = s3.magnitude();
  ASSERT_TRUE(max_shape_size(s3) == 5);

  auto ss1 = make_shape(1_c, 2_c, 3);
  tensor_shape<size_t, size_t, size_t, size_t> ss2;
  ss2 = ss1;

  ASSERT_TRUE(ss1 == ss2);

  tensor_shape<int, int, const_int<5>> shape(4);
  for (int i = 0; i < shape.magnitude(); i++) {
    invoke_with_subs(shape, i, [&](int a, int b) {
      ASSERT_TRUE(sub2ind(shape, a, b) == i);
    });
  }

  ASSERT_TRUE(cat(make_shape(1, 3, 5_c, 7_sizec), make_shape(9, 11_c)) ==
              make_shape(1, 3, 5, 7, 9, 11));
  ASSERT_TRUE(cat(cat(1_c, 3_c), make_shape(5, 7, 9_c, 11_c)) ==
              make_shape(1, 3, 5, 7, 9, 11));
  ASSERT_TRUE(cat(cat(1_c, 3_c, 5_c, 7_sizec), make_shape(9, 11)) ==
              make_shape(1, 3, 5, 7, 9, 11));

  ASSERT_TRUE(repeat_shape(make_shape(5_c), 3_c) == make_shape(5_c, 5_c, 5_c));

  static_assert(types<shape_of_rank<int, 4>>() ==
                    types<tensor_shape<int, int, int, int, int>>(),
                "");

  auto inshape = make_shape(2_c, 3_c, 4_c);
  write_tmp("shape1.cereal", inshape);
  shape_of_rank<int, 3> outshape;
  read_tmp("shape1.cereal", outshape);
  ASSERT_TRUE(inshape == outshape);
  auto inshape2 = make_shape(5_c, 4, 3, 2_c, 1);
  write_tmp("shape2.cereal", inshape2);
  tensor_shape<int, int, const_int<4>, const_int<3>, int, int> outshape2;
  read_tmp("shape2.cereal", outshape2);
  ASSERT_TRUE(inshape2 == outshape2);
}