#include <gtest/gtest.h>

#include "tensor.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;

TEST(tensor, tensor) {
  static_assert(std::is_standard_layout<mat2>::value, "");
  static_assert(!std::is_standard_layout<matx>::value, "");

  for_each([](double e) { ASSERT_EQ(e, 0.0); }, vec3());
  for_each([](double e) { ASSERT_EQ(e, 5.0); }, vec3(5, 5, 5));
  for_each([](double e) { ASSERT_EQ(e, 6.0); }, vecx(6, 6, 6));
  for_each([](double e) { ASSERT_EQ(e, 0.0); },
           cubex_<double>(make_shape(3, 4, 5)));
  for_each([](double e) { ASSERT_EQ(e, 123.0); },
           matx(make_shape(100, 100), 123));
  for_each([](double e) { ASSERT_EQ(e, 7.0); },
           matx(make_shape(2, 2), with_elements, 7, 7, 7, 7));
  for_each([](double e) { ASSERT_EQ(e, 15.0); }, vec3({15.0, 15.0, 15.0}));
  for_each([](double e) { ASSERT_EQ(e, 16.0); }, vecx({16.0, 16.0, 16.0}));
  for_each([](double e) { ASSERT_EQ(e, 17.0); },
           matx(make_shape(2, 2), {17.0, 17.0, 17.0, 17.0}));

  double data[] = {8.0, 8.0, 8.0, 8.0};
  for_each([](double e) { ASSERT_EQ(e, 8.0); },
           vec_<double, 4>(data, data + 4));
  for_each([](double e) { ASSERT_EQ(e, 8.0); }, vecx_<double>(data, data + 4));
  for_each([](double e) { ASSERT_EQ(e, 8.0); },
           matx(make_shape(2, 2), data, data + 4));

  ASSERT_TRUE(vecx(1, 2, 3).numel() == 3);
  ASSERT_TRUE(vecx(1, 2, 3, 4).numel() == 4);
  ASSERT_TRUE(vecx(1, 2, 3, 4, 5).numel() == 5);
  ASSERT_TRUE(
      (tensor<tensor_shape<size_t, const_size<2>, size_t, const_size<2>>,
              double>(1, 2, 3, 4, 5, 6, 7, 8))
          .numel() == 8);

  static_assert(tensor_of_rank<double, 5>::rank == 5, "");
}

TEST(tensor, element) {
  vecx v1(0, 1, 2, 3, 4, 5, 6, 7, 8);
  ASSERT_TRUE(v1.numel() == 9);
  for (size_t i = 0; i < 8; i++) {
    ASSERT_TRUE(v1[i] == i);
    ASSERT_TRUE(v1[first + i] == i);
    ASSERT_TRUE(v1[last + i - 8] == i);
  }
  matx m1(make_shape(3, 4), 8);
  for (size_t i = 0; i < m1.rows(); i++) {
    for (size_t j = 0; j < m1.cols(); j++) {
      ASSERT_TRUE(m1(i, j) == 8);
      ASSERT_TRUE(m1(first + i, first + j) == 8);
      ASSERT_TRUE(m1(last - i, last - j) == 8);
    }
  }
}