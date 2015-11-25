#include <gtest/gtest.h>

#include "../core/macros.hpp"
#include "../core/time.hpp"


#include "tensor_op.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;

TEST(math, tensor_constants) {
    constexpr auto t = std::make_tuple(1, 2, 3);
    constexpr auto t1 = std::get<1>(t);
    auto a = ones(make_shape(1_c, 2, 3));
    constexpr auto b = ones(1_c, 2_c, 3_c);
    auto k = a(0, 1, 2);
    cubex cube = a;
    ASSERT_TRUE(cube.size(0_c) == 1);
    ASSERT_TRUE(cube.size(1_c) == 2);
    ASSERT_TRUE(cube.size(2_c) == 3);

    for (auto & e : cube) {
        ASSERT_TRUE(e == 1);
    }

    using tt = decltype(zeros(1) + 2);
    const auto ab = a + b - a * 2;
    for (int i = 0; i < ab.numel(); i++) {
        ASSERT_EQ(ab[i], 0);
    }

    auto e = eye(2_c, 3000, 3000);
    for_each_subscript(e.shape(), [&e](auto i, auto j, auto k) {
        ASSERT_TRUE(i == j && j == k ? (e(i, j, k) == 1) : (e(i, j, k) == 0));
    });

    println(std::cout, time_cost([&e]() {
        auto sine = log(tanh(asin(sin(e)) * 2 + 5 - ones(e.shape()) / 4.0 - 3 + zeros(e.shape())));
        auto s2 = sine * 3.0 + sine / 5.0 - sine * 3.0 - sine * 0.2;
        std::cout << s2.sum() << std::endl;
        std::cout << s2.prod() << std::endl;
        std::cout << s2.norm() << std::endl;
    }));


    auto it = iota(3_c, 4_c);
    auto ite = it[1_c];
    for (int i = 0; i < it.numel(); i++) {
        ASSERT_TRUE(it[i] == i);
    }

    const auto expr = 1_symbol * ones(it.shape()) + 0_symbol;

    for (int i = 0; i < expr(5, 6).numel(); i++) {
        ASSERT_TRUE(expr(5, 6)[i] == 11);
    }

}
