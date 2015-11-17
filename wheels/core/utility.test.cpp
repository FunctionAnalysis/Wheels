#include <gtest/gtest.h>
#include "constants.hpp"
#include "utility.hpp"

using namespace wheels;

TEST(core, utility) {

    using namespace wheels::literals;
    
    auto s = conditional(false_c, 1_c, 2_c);
    static_assert(s == 2, "");
    auto s2 = conditional(true_c, 1_c, 2_c);
    static_assert(s2 == 1, "");
    constexpr int s3 = conditional(false, 1, 2);
    static_assert(s3 == 2, "");
    
    static_assert(decltype(sum(1_c, 2_c, 3_c, 4_c, 5_sizec) == 15_c)::value, "");
    static_assert(decltype(prod(1_uc, 2_c, 3_c, 4_c, 5_uc) == 120_sizec)::value, "");
    static_assert(decltype(min(5_c, 6_sizec, 34_c, 1_c, 2_uc))::value == 1, "");
    static_assert(decltype(max(5_c, 6_sizec, 34_c, 1_c, 2_uc))::value == 34, "");

    wheels::traverse([](auto v) {std::cout << v << std::endl; }, 1_c, 2_c);

}

