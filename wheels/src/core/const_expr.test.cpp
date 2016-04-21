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

struct tuple_maker {
  template <class... Ts> auto operator()(Ts &&... ts) const {
    return std::make_tuple(ts...);
  }
};

TEST(core, smart_invoke) {
  ASSERT_TRUE(smart_invoke(tuple_maker(), 1, 2) == std::make_tuple(1, 2));
  ASSERT_TRUE(smart_invoke(tuple_maker(), 1_symbol, 2,
                           0_symbol)(std::string("hahaha"), true) ==
              std::make_tuple(true, 2, std::string("hahaha")));
}