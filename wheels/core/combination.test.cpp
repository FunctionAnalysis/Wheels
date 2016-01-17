#include <gtest/gtest.h>

#include "combination.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(core, combination) {

  constexpr auto c1 = combine(1_c, 2_sizec, 3_c);
  static_assert(c1.size == 3_c, "");
  static_assert(c1[0_c] == 1_c, "");
  static_assert(c1[1_c] == 2_c, "");
  static_assert(c1[2_c] == 3_c, "");
  static_assert(std::is_empty<decltype(c1)>::value, "");

  constexpr auto c2 = combine(1, 2, 3_c);
  static_assert(c2.size == 3_c, "");
  static_assert(c2[0_c] == 1_c, "");
  static_assert(c2[1_c] == 2_c, "");
  static_assert(c2[2_c] == 3_c, "");
  static_assert(!std::is_empty<decltype(c2)>::value, "");

  constexpr int e1 = 1, e2 = 2;
  constexpr auto e3 = 3_sizec;
  constexpr auto c3 = combine(move(e1), e2, e3);
  static_assert(c3.size == 3_c, "");
  static_assert(c3[0_c] == 1_c, "");
  static_assert(c3[1_c] == 2_c, "");
  static_assert(c3[2_c] == 3_c, "");
  static_assert(!std::is_empty<decltype(c3)>::value, "");
}