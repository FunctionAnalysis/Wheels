#include "types.hpp"
#include <gtest/gtest.h>

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
}