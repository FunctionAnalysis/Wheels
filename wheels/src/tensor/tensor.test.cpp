#include <gtest/gtest.h>

#include "../../tensor"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;

TEST(tensor, tensor) {
  static_assert(std::is_standard_layout<mat2>::value, "");
  static_assert(!std::is_standard_layout<matx>::value, "");

  std::max(1, 2);

  constexpr auto sss = sizeof(vec3);

  vec3().for_each([](double e) { ASSERT_EQ(e, 0.0); });
  vec3(5, 5, 5).for_each([](double e) { ASSERT_EQ(e, 5.0); });
  vecx(6, 6, 6).for_each([](double e) { ASSERT_EQ(e, 6.0); });
  cubex_<double>(make_shape(3, 4, 5))
      .for_each([](double e) { ASSERT_EQ(e, 0.0); });
  matx(make_shape(100, 100), 123)
      .for_each([](double e) { ASSERT_EQ(e, 123.0); });
  matx(make_shape(2, 2), with_elements, 7, 7, 7, 7)
      .for_each([](double e) { ASSERT_EQ(e, 7.0); });
  vec3({15.0, 15.0, 15.0}).for_each([](double e) { ASSERT_EQ(e, 15.0); });
  vecx({16.0, 16.0, 16.0}).for_each([](double e) { ASSERT_EQ(e, 16.0); });
  matx(make_shape(2, 2), {17.0, 17.0, 17.0, 17.0})
      .for_each([](double e) { ASSERT_EQ(e, 17.0); });

  double data[] = {8.0, 8.0, 8.0, 8.0};
  vec_<double, 4>(data, data + 4).for_each([](double e) { ASSERT_EQ(e, 8.0); });
  vecx_<double>(data, data + 4).for_each([](double e) { ASSERT_EQ(e, 8.0); });

  matx(make_shape(2, 2), data, data + 4)
      .for_each([](double e) { ASSERT_EQ(e, 8.0); });

  ASSERT_TRUE(vecx(1, 2, 3).numel() == 3);
  ASSERT_TRUE(vecx(1, 2, 3, 4).numel() == 4);
  ASSERT_TRUE(vecx(1, 2, 3, 4, 5).numel() == 5);
  ASSERT_TRUE(
      (tensor<double,
              tensor_shape<size_t, const_size<2>, size_t, const_size<2>>>(
           1, 2, 3, 4, 5, 6, 7, 8))
          .numel() == 8);

  static_assert(tensor_of_rank<double, 5>::rank == 5, "");
  auto t = zeros(3, 3, 3, 3, 3).ewised().transform([](double e) { return e + 1; });
  auto tt = std::move(t).ewised().cast<int>();
  tt.eval().for_each([](int e) { ASSERT_EQ(e, 1); });
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

TEST(tensor, demo) {
  // t1: a 3x4x5 double type tensor filled with 1's
  auto t1 = ones(3, 4, 5).eval();
  // t2: a 2x2x2x2 complex<double> type tensor filled with 0's
  auto t2 = zeros<std::complex<double>>(2, 2, 2, 2).eval();
  // t3: a 3-vector, with static shape, initialized with 1, 2, 3
  vec3 t3(1, 2, 3);
  // t4: a 3-vector, with dynamic shape, initialized with 1, 2, 3
  vecx t4(1, 2, 3);
  // t5: a 5x5 matrix, with dynamic shape, filed with 5's
  matx t5(make_shape(5, 5), 5);
  // static shaped tensor types are stack allocated, and satisfy standard layout
  static_assert(sizeof(vec_<double, 3>) == 24, "");
  static_assert(std::is_standard_layout<vec_<double, 3>>::value, "");
  static_assert(sizeof(mat_<double, 2, 2>) == 32, "");
  static_assert(std::is_standard_layout<mat_<double, 2, 2>>::value, "");
}

TEST(tensor, demo2) {

  auto greeting = "hello world! 123456"_ts;
  println(greeting);
  // show only letters
  println(greeting[where('a' <= greeting && greeting <= 'z' ||
                         'A' <= greeting && greeting <= 'Z')]);

  // reverse the string
  println(greeting[last - iota(length)]);
  println(greeting[range(length, -1, -1)]);

  // concatenate the strings
  println(cat(greeting, " "_ts, "let's rock!"_ts));

  // promote the string from a vector to a matrix,
  // repeat it along rows, and transpose it
  println(repeat(promote(1_c, greeting), 3, 1).t());
}
