#include <gtest/gtest.h>
#include "tensor_functions.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, constants) {

    auto a = constants(make_shape(7, 2_c), 99.0);
    ASSERT_TRUE(std::all_of(a.begin(), a.end(), [](auto && e) {return e == 99; }));

    auto b = zeros(10_c, 10_c);
    auto eb = eval<false>(b);
    eb(5, 5) = 1;
    eb(0, 0) = 0;
    ASSERT_TRUE(eb.data_provider().indexer.size() == 2); // {1, 0}

    auto eeb = eval<false>(eb);
    ASSERT_TRUE(eeb.data_provider().indexer.size() == 1); // {1} 

}


TEST(tensor, meshgrid) {

    matx x, y;
    std::tie(x, y) = meshgrid(3, 3);


}


TEST(tensor, ewise_op_result) {

    auto a = ones(make_shape(5, 5, 5)) - 1;
    auto ea = eval<false>(a);



}
