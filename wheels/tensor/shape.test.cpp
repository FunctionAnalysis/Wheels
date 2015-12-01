#include <numeric>
#include <algorithm>
#include <gtest/gtest.h>

#include "shape.hpp"

using namespace wheels;

TEST(math, tensor_shape) {

    using namespace wheels::literals;

    auto s1 = make_shape(1_c, 2_c, 4, 5);
    std::cout << s1 << std::endl;
    auto test = s1[0_c] == 1_c && s1.at(1_c) == 2_c;
    static_assert(test, "");
    
    ASSERT_TRUE(s1.at(2_c) == 4);
    ASSERT_TRUE(s1.magnitude() == 40);

    ASSERT_TRUE(max_shape_size(s1) == 5);

    s1.resize(2_c, 5);
    ASSERT_TRUE(s1[2_c] == 5);

    ASSERT_TRUE(s1.magnitude() == 50);

    auto s2 = make_shape(cat(1_c, 2_c, 5_c, 5_c));
    std::cout << s2 << std::endl;

    ASSERT_TRUE(s1 == s2);
    ASSERT_TRUE(max_shape_size(s2) == 5);

    std::vector<size_t> inds;
    for_each_subscript(s2, [&s2, &inds](auto ... subs) {
        inds.push_back(sub2ind(s2, subs...));
    });
    std::vector<size_t> inds2(inds.size());
    std::iota(inds2.begin(), inds2.end(), 0);

    ASSERT_TRUE(inds == inds2);

    constexpr auto s3 = make_shape<size_t>(1_sizec, 5_c);
    auto m3 = s3.magnitude();
    ASSERT_TRUE(max_shape_size(s3) == 5);


    auto ss1 = make_shape(1_c, 2_c, 3);
    tensor_shape<size_t, size_t, size_t, size_t> ss2;
    ss2 = ss1;

    ASSERT_TRUE(ss1 == ss2);

    tensor_shape<int, int, const_int<5>> shape(4);
    for (int i = 0; i < shape.magnitude(); i++) {
        invoke_with_subs(shape, i, [&shape](int a, int b) {
            println(std::cout, sub2ind(shape, a, b)); 
        });
    }

}