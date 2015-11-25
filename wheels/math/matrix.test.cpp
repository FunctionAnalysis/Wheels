#include <gtest/gtest.h>

#include "matrix.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;


TEST(math, matrix) {

    auto m1 = make_rotate3(ones(3), 5.0);
    auto m = m1.m();
    auto n = m1.n();
    auto m2 = make_rotate4(ones(3).normalized(), 2.0);

    ASSERT_DOUBLE_EQ((make_rotate3(unit_x(), 2) * make_rotate3(unit_x(), 3) - make_rotate3(unit_x(), 5)).norm(), 0);
    ASSERT_DOUBLE_EQ((make_rotate3(unit_y(), 2) * make_rotate3(unit_y(), 3) - make_rotate3(unit_y(), 5)).norm(), 0);
    ASSERT_DOUBLE_EQ((make_rotate3(unit_z(), 2) * make_rotate3(unit_z(), 3) - make_rotate3(unit_z(), 5)).norm(), 0);

    ASSERT_DOUBLE_EQ((make_rotate4(unit_x(), 2) * make_rotate4(unit_x(), 3) - make_rotate4(unit_x(), 5)).norm(), 0);
    ASSERT_DOUBLE_EQ((make_rotate4(unit_y(), 2) * make_rotate4(unit_y(), 3) - make_rotate4(unit_y(), 5)).norm(), 0);
    ASSERT_DOUBLE_EQ((make_rotate4(unit_z(), 2) * make_rotate4(unit_z(), 3) - make_rotate4(unit_z(), 5)).norm(), 0);

    for (double angle : {0.2, 0.5, 1.0, 1.2}) {
        auto a = unit_y();
        auto b = make_rotate3(unit_x(), angle) * a;

        double dotv = dot(a, b);
        ASSERT_DOUBLE_EQ(acos(abs(dotv)), angle);
    }

    matx m3 = make_look_at(zeros(3), unit_x(3_c), unit_y(3));

}
