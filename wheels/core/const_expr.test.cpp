#include <gtest/gtest.h>
#include "const_expr.hpp"

using namespace wheels;

TEST(core, const_expr) {

    using namespace wheels::literals;

    constexpr auto n1 = 0_symbol;
    constexpr auto nn1 = -n1;

    constexpr auto n2 = 1_symbol;
    constexpr auto sum = - n1 + n2 * (n1 % 3_c) + 1_c - 5_c + 1;

    constexpr auto sumv = sum(5_c, 5_c);
    static_assert(sumv == (-5 + 5 * (5 % 3) + 1 - 5) + 1, "");

}