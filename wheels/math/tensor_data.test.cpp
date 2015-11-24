#include <gtest/gtest.h>

#include "tensor_data.hpp"

using namespace wheels;

TEST(math, test) {

    using namespace wheels::literals;
    
    auto shape = make_shape<int>(1, 2_c, 3, 4_c);

    static_assert(tdp::is_constructible_with_shape<std::array<double, 5>>::value, "");
    static_assert(tdp::is_constructible_with_elements<std::array<double, 5>>::value, "");
    static_assert(tdp::is_constructible_with_shape_elements<std::array<double, 5>>::value, "");



}