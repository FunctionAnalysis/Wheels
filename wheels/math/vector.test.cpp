#include <gtest/gtest.h>
#include "vector.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;


TEST(math, vector) {

    auto l = ones(3).length();
    auto v = ones(3).normalized();
    auto vv = v.normalized();
    
    auto x = unit_x();
    ASSERT_EQ(x.length(), 1);
    ASSERT_EQ(x.x(), 1);
    ASSERT_EQ(x.y(), 0);
    ASSERT_EQ(x.z(), 0);
    ASSERT_EQ(x.red(), 1);
    ASSERT_EQ(x.green(), 0);
    ASSERT_EQ(x.blue(), 0);

    auto y = unit_y();
    ASSERT_EQ(y.length(), 1);
    ASSERT_EQ(y.x(), 0);
    ASSERT_EQ(y.y(), 1);
    ASSERT_EQ(y.z(), 0);
    ASSERT_EQ(y.red(), 0);
    ASSERT_EQ(y.green(), 1);
    ASSERT_EQ(y.blue(), 0);

    auto z = unit_z();
    ASSERT_EQ(z.x(), 0);
    ASSERT_EQ(z.y(), 0);
    ASSERT_EQ(z.z(), 1);
    ASSERT_EQ(z.red(), 0);
    ASSERT_EQ(z.green(), 0);
    ASSERT_EQ(z.blue(), 1);

    auto x4 = unit_x(4_sizec);

}
