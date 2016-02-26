#include <gtest/gtest.h>

#include "const_expr.hpp"
#include "types.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(core, const_exprs) {
  constexpr auto n1 = 0_symbol;
  constexpr auto nn1 = -n1;

  constexpr auto n2 = 1_symbol;
  constexpr auto sum = -n1 + n2 * (n1 % 3_c) + 1_c - 5_c + 1_c;

  auto sumv = sum(5_c, 5_c);
  static_assert(sumv == (-5 + 5 * (5 % 3) + 1 - 5) + 1, "");
}

struct print_types {
  template <class... Ts> void operator()(Ts &&... ts) const {
    println(type_of(ts)...);
  }
};

TEST(core, smart_invoke) {
  auto e = details::_has_const_expr(1_symbol);
  smart_invoke(print_types(), 1, 2);
  smart_invoke(print_types(), 1_symbol, 2, 0_symbol)("hahaha", true);
}