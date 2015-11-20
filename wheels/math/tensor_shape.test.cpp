#include <numeric>
#include <algorithm>
#include <gtest/gtest.h>
#include "tensor_shape.hpp"

using namespace wheels;

TEST(math, tensor_shape) {

    using namespace wheels::literals;

    auto s1 = make_tensor_shape<int>(1_c, 2_c, 4, 5);
    std::cout << s1 << std::endl;
    auto test = s1[0_c] == 1_c && s1.at(1_c) == 2_c;
    static_assert(test, "");
    
    ASSERT_TRUE(s1.at(2_c) == 4);
    ASSERT_TRUE(s1.magnitude() == 40);

    s1.resize(2_c, 5);
    ASSERT_TRUE(s1[2_c] == 5);

    ASSERT_TRUE(s1.magnitude() == 50);

    auto s2 = make_tensor_shape(cat(1_c, 2_c, 5_c, 5_c));
    std::cout << s2 << std::endl;

    ASSERT_TRUE(s1 == s2);

    std::vector<size_t> inds;
    s2.for_each_subscript([&s2, &inds](auto ... subs) {
        inds.push_back(s2.sub2ind(subs...));
    });
    std::vector<size_t> inds2(inds.size());
    std::iota(inds2.begin(), inds2.end(), 0);

    ASSERT_TRUE(inds == inds2);

    constexpr auto s3 = make_tensor_shape<size_t>(1_sizec, 5_c);
    auto m3 = s3.magnitude();


    auto ss1 = make_tensor_shape<int>(1_c, 2_c, 3);
    tensor_shape<size_t, size_t, size_t, size_t> ss2;
    ss2 = ss1;

    ASSERT_TRUE(ss1 == ss2);

}


void test(const tensor_shape<int, int> & s) restrict(amp) {

    int ind = 5;
    int sub = 0;
    s.ind2sub(ind, sub);

}
