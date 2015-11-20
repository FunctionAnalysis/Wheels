#include <gtest/gtest.h>

#include "constants.hpp"

using namespace wheels;

TEST(core, constants) {

    using namespace wheels::literals;

    auto test1 = 1_c != 2_uc && 1_c == 1_uc;
    auto test2 = !!test1;
    static_assert(test1, "");
    static_assert(test2, "");

    auto b3 = 1_c + 2_uc == 3_c;
    auto b4 = 5_c - 4_uc == 1_c;
    auto b5 = 10_c * 100_c == 1000_c;
    auto b6 = 100_c / 10_c == 10_c;
    auto b7 = 101_c % 10_c == 1_uc;
    static_assert(b3 && b4 && b5 && b6 && b7, "");

    auto seq1 = cat(1_c, 2_c, 5_c);
    static_assert(seq1[0_uc] == 1_c, "");
    static_assert(seq1[1_uc] == 2_c, "");
    static_assert(seq1[2_uc] == 5_c, "");
    static_assert((-seq1 * 5_c + 77_c == cat(-5_c, -10_c, -25_c) + 77_c).all(), "");
    static_assert(cat(seq1, seq1)[3_c] == 1_c, "");

    ASSERT_EQ(1_c * 5, 5);

}


template <char C>
void foo() restrict(amp) {

}

void test() restrict(amp) {

    using namespace wheels::literals;

    constexpr const_ints<int, 8> a;
    const_ints<int, 1, 2, 3>::sum();
    const_ints<int, 1, 2, 3> aa;
    constexpr bool iszero = aa.length == 0;
    aa[const_ints<int, 2>()];

    constexpr auto s = make_const_sequence(const_int<10>());
    constexpr auto r = make_const_range(const_int<3>(), const_int<8>());

    constexpr auto s_r = cat(s, r);

    constexpr auto v = conditional(yes(), const_int<1>(), const_int<2>());
    constexpr auto v2 = conditional(no(), const_int<1>(), const_int<2>());
    constexpr auto v3 = v2 * (v2 + v);

}