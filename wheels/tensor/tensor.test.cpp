#include <gtest/gtest.h>

#include "methods.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;

TEST(tensor, tensor) {
  static_assert(std::is_standard_layout<mat2>::value, "");
  static_assert(!std::is_standard_layout<matx>::value, "");

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
      (tensor<tensor_shape<size_t, const_size<2>, size_t, const_size<2>>,
              double>(1, 2, 3, 4, 5, 6, 7, 8))
          .numel() == 8);

  static_assert(tensor_of_rank<double, 5>::rank == 5, "");
  auto t = zeros(3, 3, 3, 3, 3).transform([](double e) { return e + 1; });
  auto tt = static_ecast<int>(std::move(t));
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

TEST(tensor, serialize) {
  write_tmp("vec3.cereal", vec3(1, 2, 3));
  vec3 v;
  read_tmp("vec3.cereal", v);
  ASSERT_TRUE(v == vec3(1, 2, 3));
  vecx v2;
  read_tmp("vec3.cereal", v2);
  ASSERT_TRUE(v2 == vec3(1, 2, 3));

  auto vb = ones<bool>(5).eval();
  write_tmp("vecb5", vb);
  vecx_<bool> vb2;
  read_tmp("vecb5", vb2);
  ASSERT_TRUE(vb == vb2);
}


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

  auto rre = rr.eval().t().t().t().t();
  static_assert(types<decltype(rre)>() == types<decltype(rr.eval())>(), "");
  ASSERT_TRUE(rre == rr);

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

TEST(tensor, methods3) {
  auto fun = max(0_symbol + 1, 1_symbol * 2);
  auto result1 = fun(3, 2); // 0_symbol->3, 1_symbol->2, result1 = 4 of int
  ASSERT_EQ(result1, 4);
  auto result2 = fun(vec3(2, 3, 4), ones(3) * 2); // 0_symbol->vec3(2, 3, 4),
  auto e0 = element_at(result2, 0);
  auto e1 = element_at(result2, 1);
  auto e2 = element_at(result2, 2);
  auto t = result2.eval();
  ASSERT_TRUE(result2 == vec3(4, 4, 5));
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

TEST(tensor, permute) {
  auto t = zeros(1, 2, 3, 4, 5).eval();
  std::default_random_engine rng;
  randomize_fields(t, rng);
  ASSERT_TRUE(t != zeros(t.shape()));
  auto permuted = permute(t, 2_c, 4_c, 0_c, 3_c, 1_c).eval();
  for_each_subscript(
      t.shape(), [&t, &permuted](auto s0, auto s1, auto s2, auto s3, auto s4) {
        ASSERT_EQ(t(s0, s1, s2, s3, s4), permuted(s2, s4, s0, s3, s1));
      });
}