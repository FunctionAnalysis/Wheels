#include <gtest/gtest.h>

#include "tensor_op.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;

TEST(math, tensor_constants) {
    constexpr auto t = std::make_tuple(1, 2, 3);
    constexpr auto t1 = std::get<1>(t);
    constexpr auto a = ones(make_shape(1_c, 2_c, 3_c));
    constexpr auto b = ones(1_c, 2_c, 3_c);
    auto k = a(0, 1, 2);
    cubex cube = a;
    ASSERT_TRUE(cube.size(0_c) == 1);
    ASSERT_TRUE(cube.size(1_c) == 2);
    ASSERT_TRUE(cube.size(2_c) == 3);

    for (auto & e : cube.storage()) {
        ASSERT_TRUE(e == 1);
    }

    const auto ab = a + b - a * 2;
    for (int i = 0; i < ab.numel(); i++) {
        ASSERT_EQ(ab[i], 0);
    }

    constexpr auto e = eye(2_c, 3_c, 4_c);
    for_each_subscript(e.shape(), [&e](auto i, auto j, auto k) {
        ASSERT_TRUE(i == j && j == k ? (e(i, j, k) == 1) : (e(i, j, k) == 0));
    });

}
