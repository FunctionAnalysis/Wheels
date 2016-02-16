#include <gtest/gtest.h>

#include "../tensor.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;

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
  ASSERT_EQ(result1, max(4, 4));
  auto result2 = fun(vec3(2, 3, 4), ones(3) * 2); // 0_symbol->vec3(2, 3, 4),
  auto e0 = element_at(result2, 0ull);
  auto e1 = element_at(result2, 1ull);
  auto e2 = element_at(result2, 2ull);
  auto f0 = result2(0ull);
  auto f1 = result2(1ull);
  auto f2 = result2(2ull);
  ASSERT_TRUE(result2 == vec3(4, 4, 5));
  std::cout << result2 << std::endl;
  auto t = result2.eval();
  std::cout << t << std::endl;
  //ASSERT_TRUE(t == vec3(4, 4, 5));
}