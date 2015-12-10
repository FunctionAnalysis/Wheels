#include "constants.hpp"
#include "utility.hpp"
#include <gtest/gtest.h>

using namespace wheels;

TEST(core, utility) {

  using namespace wheels::literals;

  auto s = conditional(false_c, 1_c, 2_c);
  static_assert(s == 2, "");
  auto s2 = conditional(true_c, 1_c, 2_c);
  static_assert(s2 == 1, "");
  constexpr int s3 = conditional(false, 1, 2);
  static_assert(s3 == 2, "");

  static_assert(sum(1_c, 2_c, 3_c, 4_c, 5_sizec) == 15_c, "");
  static_assert(prod(1_uc, 2_c, 3_c, 4_c, 5_uc) == 120_sizec, "");
  static_assert(min(5_c, 6_sizec, 34_c, 1_c, 2_uc) == 1, "");
  static_assert(max(5_c, 6_sizec, 34_c, 1_c, 2_uc) == 34, "");

  static_assert(all_same(1_c, 1, 1_c), "");
  static_assert(!all_same(1_c, 2, 1_c), "");
  static_assert(all_same(1_c, 1, 1_c, 1, 1, 1_sizec), "");

  static_assert(all_different(1_c, 2_c, 3_c, 4_c, 5_indexc), "");
  static_assert(!all_different(1_c, 4_c, 3_c, 4_c, 5_indexc), "");

  static_assert(make_ordered_pair(5_c, 4_c) == std::make_pair(4_c, 5_c), "");
  static_assert(make_ordered_pair(4_c, 5_c) == std::make_pair(4_c, 5_c), "");

  static_assert(bounded(0_c, 3_c, 5_c) == 3_c, "");
  static_assert(bounded(4_c, 3_c, 5_c) == 4_c, "");
  static_assert(bounded(6_c, 3_c, 5_c) == 5_c, "");
  static_assert(is_between(5_c, 0_c, 6_c), "");
  static_assert(!is_between(5_c, 0_c, 5_c), "");
  static_assert(is_between(0_c, 0_c, 5_c), "");

  wheels::traverse([](auto v) { std::cout << v << std::endl; }, 1_c, 2_c);
}
