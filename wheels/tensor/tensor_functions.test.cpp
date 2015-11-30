#include <gtest/gtest.h>

#include <complex>

#include "tensor_functions.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace std::complex_literals;

TEST(tensor, constants) {

    auto a = constants(make_shape(7, 2_c), 99.0);
    ASSERT_TRUE(std::all_of(a.begin(), a.end(), [](auto && e) {return e == 99; }));

    auto b = zeros(10_c, 10_c);
    auto eb = eval<false>(b);
    eb(5, 5) = 1;
    eb(0, 0) = 0;
    ASSERT_TRUE(eb.data_provider().size() == 2); // {1, 0}

    auto eeb = eval<false>(eb);
    ASSERT_TRUE(eeb.data_provider().size() == 1); // {1} 

}


TEST(tensor, meshgrid) {

    matx x, y;
    std::tie(x, y) = meshgrid(100, 100);

    for_each_subscript(x.shape(), [&x, &y](size_t s1, size_t s2) {
        ASSERT_TRUE(x(s1, s2) == s1);
        ASSERT_TRUE(y(s1, s2) == s2);
    });

    spmat x2, y2;
    std::tie(x2, y2) = meshgrid(make_shape(100_c, 100_c));

    ASSERT_TRUE(x + y * 1i == x2 + y2 * 1i);

}


TEST(tensor, ewise_op_result) {

    auto a = ones(make_shape(5, 5, 5)) - 1;
    auto ea = eval<false>(a);



}
