#include <gtest/gtest.h>

#include "const_expr.hpp"

TEST(core, const_symbols) {

    using namespace wheels::literals;

    constexpr auto n1 = 0_symbol;
    constexpr auto nn1 = -n1;

    constexpr auto n2 = 1_symbol;
    constexpr auto sum = -n1 + n2 * (n1 % 3_c) + 1_c - 5_c + 1_c;

    auto sumv = sum(5_c, 5_c);
    static_assert(sumv == (-5 + 5 * (5 % 3) + 1 - 5) + 1, "");


}