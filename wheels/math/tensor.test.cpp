#include <gtest/gtest.h>

#include "tensor.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;

static_assert(std::is_standard_layout<mat2x2>::value, "");
static_assert(std::is_standard_layout<mat2x3>::value, "");

struct A { int a; };
struct B : A {};
struct C : B {};
static_assert(std::is_standard_layout<C>::value, "");


TEST(math, test1) {
    mat2x2 m1;
    mat2x2 m2(with_elements, 1, 2, 3, 4);
    matx m3 = m2;
    ASSERT_TRUE(m3.numel() == 4);
    ASSERT_TRUE(m2[0] == 1);
    ASSERT_TRUE(m2(last, last) == 4);
    ASSERT_TRUE(m3[0] == 1);
    ASSERT_TRUE(m3(last, last) == 4);
}

TEST(math, tensor_dynamic) {
    matx m1;
    matx m2(make_shape(2, 2));
    matx m3(make_shape(2_c, 2), with_elements, 1, 2, 3, 4);
}