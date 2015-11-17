#include <gtest/gtest.h>
#include "tensor.hpp"

using namespace wheels;

TEST(math, test) {

    std::vector<int> v(100, 0);
    println(std::cout, "capacity: ", v.capacity());
    v.resize(101);
    println(std::cout, "capacity: ", v.capacity());

    constexpr std::array<int, 3> arr = { 1, 2, 3 };

    using namespace wheels::index_tags;

    vec3 v3 = { 1, 2, 3 };
    auto e = v3[0];
    ASSERT_TRUE(v3[0] == 1);
    ASSERT_TRUE(v3[last] == 3);

    mat2 m2 = { 1, 2, 3, 4 };
    ASSERT_TRUE(m2(first, 0) == 1);
    ASSERT_TRUE(m2[last] == 4);

    vecx vx = { 1, 2, 3, 4, 5 };


    //vecx_gpu vx_gpu;

}