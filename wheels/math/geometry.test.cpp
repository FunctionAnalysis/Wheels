#include <gtest/gtest.h>
#include "geometry.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;


TEST(math, matrix) {
    
    auto l = ones(3).length();
    auto v = ones(3).normalized();
    auto vv = v.normalized();
    auto m1 = make_rotate3(ones(3), 5.0);
    auto m = m1.m();
    auto n = m1.n();
    auto m2 = make_rotate4(ones(3).normalized(), 2.0);
    
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

    ASSERT_DOUBLE_EQ((make_rotate3(unit_x(), 2) * make_rotate3(unit_x(), 3) - make_rotate3(unit_x(), 5)).norm(), 0);
    ASSERT_DOUBLE_EQ((make_rotate3(unit_y(), 2) * make_rotate3(unit_y(), 3) - make_rotate3(unit_y(), 5)).norm(), 0);
    ASSERT_DOUBLE_EQ((make_rotate3(unit_z(), 2) * make_rotate3(unit_z(), 3) - make_rotate3(unit_z(), 5)).norm(), 0);

    ASSERT_DOUBLE_EQ((make_rotate4(unit_x(), 2) * make_rotate4(unit_x(), 3) - make_rotate4(unit_x(), 5)).norm(), 0);
    ASSERT_DOUBLE_EQ((make_rotate4(unit_y(), 2) * make_rotate4(unit_y(), 3) - make_rotate4(unit_y(), 5)).norm(), 0);
    ASSERT_DOUBLE_EQ((make_rotate4(unit_z(), 2) * make_rotate4(unit_z(), 3) - make_rotate4(unit_z(), 5)).norm(), 0);

    for (double angle : {0.2, 0.5, 1.0, 1.2}) {
        auto a = unit_y();
        auto b = make_rotate3(unit_x(), angle) * a;

        double dotv = ewise_mul(a, b).sum();
        ASSERT_DOUBLE_EQ(acos(abs(dotv)), angle);
    }
}
