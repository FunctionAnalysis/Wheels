#include <gtest/gtest.h>
#include "constants.hpp"
#include "utility.hpp"

using namespace wheels;

TEST(core, utility) {

    using namespace wheels::literals;
    
    auto s = conditional(false_c, 1_c, 2_c);
    auto s2 = conditional(true_c, 1_c, 2_c);
    constexpr int s3 = conditional(false, 1, 2);
    
    auto b1 = sum(1_c, 2_c, 3_c, 4_c, 5_sizec) == 15_c;
    auto b2 = prod(1_uc, 2_c, 3_c, 4_c, 5_uc) == 120_sizec;
    static_assert(b1, "");
    static_assert(b2, "");

    auto b3 = min(5_c, 6_sizec, 34_c, 1_c, 2_uc);
    static_assert(b3.value == 1, "");

    auto b4 = max(5_c, 6_sizec, 34_c, 1_c, 2_uc);
    static_assert(b4.value == 34, "");

    wheels::traverse([](auto v) {std::cout << v << std::endl; }, 1_c, 2_c);

}

