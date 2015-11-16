#include <gtest/gtest.h>
#include "constants.hpp"

using namespace wheels;

TEST(core, constants) {

    constexpr int a = 1;
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
    auto test3 = (b3 && b4 && b5 && b6 && b7);
    static_assert(test3, "");

    //times(5_c, []() {std::cout << "repeat 5 times" << std::endl; });

    auto seq1 = cat(1_c, 2_c, 5_c);

    auto b8 = (-seq1 * 5_c == cat(-5_c, -10_c, -25_c)).all();
    static_assert(b8, "");




}