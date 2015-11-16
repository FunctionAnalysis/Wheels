#include <gtest/gtest.h>
#include "tensor_shape.hpp"

using namespace wheels;

TEST(math, tensor_shape) {

    using namespace wheels::literals;

    auto s1 = make_tensor_shape(1_c, 2_c, 4, 5);
    auto test = s1.at(0_c) == 1_c && s1.at(1_c) == 2_c;
    static_assert(test, "");
    
    ASSERT_TRUE(s1.at(2_c) == 4);
    ASSERT_TRUE(s1.magnitude() == 40);

    s1.resize(2_c, 5);
    ASSERT_TRUE(s1.at(2_c) == 5);

    ASSERT_TRUE(s1.magnitude() == 50);

    auto s2 = make_tensor_shape(cat(1_c, 2_c, 5_c, 5_c));
    ASSERT_TRUE(s1 == s2);



}