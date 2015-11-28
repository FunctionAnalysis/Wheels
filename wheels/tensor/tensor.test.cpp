#include <gtest/gtest.h>

#include <complex>
#include "tensor.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;

TEST(tensor, fix_sized){

    static_assert(std::is_standard_layout<vec3>::value, "");
    static_assert(std::is_standard_layout<vec4>::value, "");
    static_assert(std::is_standard_layout<mat3x3>::value, "");
    
    constexpr vec3 v3(1, 2, 3);

    ASSERT_EQ(v3[0], 1);
    ASSERT_EQ(v3[first], 1);
    ASSERT_EQ(v3[1], 2);
    ASSERT_EQ(v3[2], 3);
    ASSERT_EQ(v3[last], 3);
    ASSERT_EQ(v3[first + 1], 2);
    ASSERT_EQ(v3[first + 1_c], 2);
    ASSERT_EQ(v3[last - 1], 2);
    ASSERT_EQ(v3[length / 2], 2);

    constexpr vec4 v4(1, 2, 3);

    ASSERT_EQ(v4[last], 0);

    constexpr mat3x3 m33(1, 2, 3, 4, 5, 6, 7, 8, 9);
    ASSERT_EQ(m33(0, 0), 1);
    ASSERT_EQ(m33(first, first), 1);
    ASSERT_EQ(m33(last, last), 9);

    ASSERT_EQ(std::accumulate(m33.begin(), m33.end(), 0.0), 45);
    ASSERT_EQ(std::accumulate(m33.nzbegin(), m33.nzend(), 0.0), 45);

}


TEST(tensor, dynamic_sized) {

    vecx v5(1, 2, 3, 4, 5);

    ASSERT_EQ(v5[0], 1);
    ASSERT_EQ(v5[first], 1);
    ASSERT_EQ(v5[1], 2);
    ASSERT_EQ(v5[2], 3);
    ASSERT_EQ(v5[last], 5);
    ASSERT_EQ(v5[first + 1], 2);
    ASSERT_EQ(v5[first + 1_c], 2);
    ASSERT_EQ(v5[last - 1], 4);
    ASSERT_EQ(v5[length / 2], 3);

    vec3 sv3(1, 2, 3);
    vecx v3 = sv3;

    ASSERT_EQ(v3[0], 1);
    ASSERT_EQ(v3[first], 1);
    ASSERT_EQ(v3[1], 2);
    ASSERT_EQ(v3[2], 3);
    ASSERT_EQ(v3[last], 3);
    ASSERT_EQ(v3[first + 1], 2);
    ASSERT_EQ(v3[first + 1_c], 2);
    ASSERT_EQ(v3[last - 1], 2);
    ASSERT_EQ(v3[length / 2], 2);

    mat3x2 sm32(6, 5, 4, 3, 2, 1);
    matx m32 = sm32;

    ASSERT_EQ(m32(0, 0), 6);
    ASSERT_EQ(m32(first, first), 6);
    ASSERT_EQ(m32(last, last), 1);

    ASSERT_EQ(std::accumulate(m32.nzbegin(), m32.nzend(), 0.0), 21);

}


TEST(tensor, sparse) {

    spvec va(make_shape(100));
    va[2] = 2;
    va[99] = 99;
    
    ASSERT_EQ(std::accumulate(va.nzbegin(), va.nzend(), 0.0), 101);

    spvec_<std::complex<double>> vb;
    vb = va;

    ASSERT_TRUE(vb[2] == 2.0);
    ASSERT_TRUE(vb[99] == 99.0);

}
