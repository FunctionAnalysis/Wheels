#include <gtest/gtest.h>

#include "ewise.hpp"

#include "constants.hpp"
#include "permute.hpp"
#include "block.hpp"
#include "iota.hpp"
#include "tensor.hpp"
#include "matrix.hpp"
#include "diagonal.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::tags;

TEST(tensor, ewise_ops1) {
  ASSERT_TRUE(cube2().ewised() * cube2() == cube2());
  ASSERT_TRUE(ones(50, 50).ewised() * zeros(50, 50) == zeros(50, 50));
  auto t1 = zeros(100, 100, 100);
  auto t2 = t1 + 1;
  ASSERT_TRUE(t2 == ones(100, 100, 100));

  auto a = ones(50, 30);
  auto b = a.t().t() + 1;
  ASSERT_TRUE(b == a * 2);
}

TEST(tensor, ewise_ops2) {
  auto t1 = ones(10, 100).eval();
  auto r1 = sin(t1);
  auto r2 = r1 + t1 * 2.0;
  auto rr = min(t1, r2);

  rr.for_each([](double e) { ASSERT_EQ(e, min(1.0, sin(1) + 2)); });
  rr.eval().for_each([](double e) { ASSERT_EQ(e, min(1.0, sin(1) + 2)); });
  rr.t().for_each([](double e) { ASSERT_EQ(e, min(1.0, sin(1) + 2)); });
  rr.t().t().for_each([](double e) { ASSERT_EQ(e, min(1.0, sin(1) + 2)); });

  decltype(auto) rre = rr.t().t().t().t();
  ASSERT_TRUE(&rr == &rre);
  ASSERT_TRUE(rr == rre);

  // element retreival
  auto efirst = rr[0]; // via vectoized index
  ASSERT_EQ(efirst, min(1.0, sin(1) + 2));
  auto efirst2 = rr(0, 0); // via tensor subscripts
  ASSERT_EQ(efirst2, min(1.0, sin(1) + 2));
  // index tags can be used to represent sizes
  using namespace wheels::tags;
  auto e1 = rr[length - 1]; // same with rr[100*200-1]
  auto e2 =
      rr(length / 2, (length - 20) / 2);  // same with rr(100/2, (200-20)/2)
  auto e3 = rr(9, (length / 10 + 2) * 2); // same with rr(10, (200/10+2)*2)
  auto e4 = rr(last, last / 3);           // last = length-1
  ASSERT_TRUE((vecx({e1, e2, e3, e4}).ewised().equals(efirst).all()));
}

TEST(tensor, ewise_ops3) {
  auto fun = max(0_arg + 1, 1_arg * 2);
  auto result1 = fun(3, 2); // 0_arg->3, 1_arg->2, result1 = 4 of int
  ASSERT_EQ(result1, max(4, 4));
  auto result2 = fun(vec3(2, 3, 4).ewised(), ones(3).ewised() * 2); // 0_arg->vec3(2, 3, 4),
  ASSERT_TRUE(result2 == vec3(4, 4, 5));
  std::cout << result2 << std::endl;
  auto t = result2.eval();
  std::cout << t << std::endl;
  ASSERT_TRUE(t == vec3(4, 4, 5));
}

TEST(tensor, ewise_ops4) {
  std::default_random_engine rng;
  matx_<mat3> mat_of_mats1 = rand<mat3>(make_shape(3, 3), rng);
  matx_<mat3> mat_of_mats2 = rand<mat3>(make_shape(3, 3), rng);
  auto sum1 = eval(mat_of_mats1 + mat_of_mats2);
  auto sum2 = eval(mat_of_mats1.scalarized() + mat_of_mats2);
  auto sum3 = eval(mat_of_mats1 + mat_of_mats2.scalarized());
  ASSERT_TRUE(sum1.shape() == make_shape(3, 3));
  ASSERT_TRUE(sum2.shape() == make_shape(3, 3));
  ASSERT_TRUE(sum3.shape() == make_shape(3, 3));
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      ASSERT_TRUE(sum1(i, j) == mat_of_mats1(i, j) + mat_of_mats2(i, j));
      ASSERT_TRUE(sum2(i, j) == mat_of_mats1 + mat_of_mats2(i, j));
      ASSERT_TRUE(sum3(i, j) == mat_of_mats1(i, j) + mat_of_mats2);
    }
  }
}