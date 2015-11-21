#include <gtest/gtest.h>

#include "tensor.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;

using mat2 = tensor_layout<tensor_shape<int, const_int<2>, const_int<2>>, std::array<double, 4>, platform_cpu>;
using matx = tensor_layout<tensor_shape<int, int, int>, std::vector<double>, platform_cpu>;

static_assert(std::is_standard_layout<mat2>::value, "");

struct A { int a; };
struct B : A {};
struct C : B {};
static_assert(std::is_standard_layout<C>::value, "");

constexpr int v2 = sizeof(mat2);
constexpr int vx = sizeof(matx);

TEST(math, tensor_static) {
    mat2 m1;
    mat2 m2(with_elements, 1, 2, 3, 4);
    ASSERT_TRUE(m2[0] == 1);
    ASSERT_TRUE(m2(last, last - 1) == 4);
}

TEST(math, tensor_dynamic) {
    matx m1;
    matx m2(make_shape(2, 2));
    matx m3(make_shape(2_c, 2), with_elements, 1, 2, 3, 4);
}