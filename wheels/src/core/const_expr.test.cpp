#include <gtest/gtest.h>

#include "const_expr.hpp"
#include "smart_invoke.hpp"
#include "types.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(core, const_exprs) {
  constexpr auto n1 = 0_symbol;
  auto nn1 = -n1;

  constexpr auto n2 = 1_symbol;
  auto sum = -n1 + n2 * (n1 % 3_c) + 1_c - 5_c + 1_c;

  auto sumv = sum(5_c, 5_c);
  static_assert(sumv == (-5 + 5 * (5 % 3) + 1 - 5) + 1, "");
}