#include <gtest/gtest.h>

#include "methods.hpp"

using namespace wheels;

TEST(tensor, methods) {
  auto kk = ewise_mul(cube2(), cube2()).eval();
  auto kk2 = ewise_mul(ones(50, 50), zeros(50, 50)).eval();
  auto t1 = zeros(100, 100, 100);
  auto t2 = t1 + 1;
  auto &k = t2[3];
  ASSERT_TRUE(t2 == ones(100, 100, 100));

  auto a = ones(50, 30);
  auto b = a.t().t() + 1;
}

TEST(tensor, methods2) {
  auto t1 = ones(10, 100).eval();
  auto r1 = sin(t1);
  auto r2 = r1 + t1 * 2.0;
  auto rr = min(t1, r2);

  rr.for_each([](double e) { ASSERT_EQ(e, min(1.0, sin(1) + 2)); });
  rr.eval().for_each([](double e) { ASSERT_EQ(e, min(1.0, sin(1) + 2)); });
  rr.t().for_each([](double e) { ASSERT_EQ(e, min(1.0, sin(1) + 2)); });
  rr.t().t().for_each([](double e) { ASSERT_EQ(e, min(1.0, sin(1) + 2)); });

  // element retreival
  auto efirst = rr[0];     // via vectoized index
  auto efirst2 = rr(0, 0); // via tensor subscripts
  // index tags can be used to represent sizes
  using namespace wheels::index_tags;
  auto e1 = rr[length - 1]; // same with rr[100*200-1]
  auto e2 =
      rr(length / 2, (length - 20) / 2);   // same with rr(100/2, (200-20)/2)
  auto e3 = rr(9, (length / 10 + 2) * 2); // same with rr(10, (200/10+2)*2)
  auto e4 = rr(last, last / 3);            // last = length-1
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

  static_assert(sizeof(vecx(1)) == 48, "");
  static_assert(sizeof(vecx(1, 2)) == 48, "");
  static_assert(sizeof(vecx(1, 2, 3)) == 48, "");
  // static_assert(sizeof(ones(100).eval()) == 48, "");
}