#include <gtest/gtest.h>

#include "types.hpp"

using namespace wheels;

TEST(core, type) {
  using namespace wheels::literals;

  int a = 0, b = 0;
  double c = 0;
  std::string d = "";
  auto t = type_of(a, b, c, d);

  static_assert(t[0_c].is<int &>() && t[1_c].is<int &>() &&
                    t[2_c].is<double &>() && t[3_c].is<std::string &>(),
                "");

  int && i = 1;
  decltype(auto) j = wheels_forward(i);
  using tt = decltype(i);
  int kk = 0;
  int & k = kk;
  decltype(auto) kd = wheels_forward(k);

  double ddddd = cast<by_smart_static, double>(1_c);
}
