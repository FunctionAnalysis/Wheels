#include <gtest/gtest.h>

#include <complex>

#include "tensor_categories.hpp"
#include "tensor_functions.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;

template <class CategoryT, bool Writable>
void foo(const vector<CategoryT, Writable> & v) {
   
}

TEST(tensor, fix_sized){

    static_assert(std::is_standard_layout<vec3>::value, "");
    static_assert(std::is_standard_layout<vec4>::value, "");
    static_assert(std::is_standard_layout<mat3x3>::value, "");
    
    constexpr vec3 v3(1, 2, 3);
    std::cout << v3 << std::endl;

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
    std::cout << m33 << std::endl;

    ASSERT_EQ(m33(0, 0), 1);
    ASSERT_EQ(m33(first, first), 1);
    ASSERT_EQ(m33(last, last), 9);

    ASSERT_EQ(std::accumulate(m33.begin(), m33.end(), 0.0), 45);
    ASSERT_EQ(std::accumulate(m33.nzbegin(), m33.nzend(), 0.0), 45);

    ASSERT_DOUBLE_EQ(v3.normalized().norm(), 1);

}


TEST(tensor, dynamic_sized) {

    vecx v5(1, 2, 3, 4, 5);
    std::cout << v5 << std::endl;

    ASSERT_EQ(v5[0], 1);
    ASSERT_EQ(v5[first], 1);
    ASSERT_EQ(v5[1], 2);
    ASSERT_EQ(v5[2], 3);
    ASSERT_EQ(v5[last], 5);
    ASSERT_EQ(v5[first + 1], 2);
    ASSERT_EQ(v5[first + 1_c], 2);
    ASSERT_EQ(v5[last - 1], 4);
    ASSERT_EQ(v5[length / 2], 3);

    ASSERT_DOUBLE_EQ(v5.normalized().norm(), 1);

    vec3 sv3(1, 2, 3);
    vecx v3 = sv3;

    ASSERT_EQ(v3[0], 1);
    ASSERT_EQ(v3.x(), 1);
    ASSERT_EQ(v3[first], 1);
    ASSERT_EQ(v3[1], 2);
    ASSERT_EQ(v3.y(), 2);
    ASSERT_EQ(v3[2], 3);
    ASSERT_EQ(v3.z(), 3);
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
    ASSERT_EQ(m32.sum(), 21);

    ASSERT_EQ(m32.rows(), 3);
    ASSERT_EQ(m32.cols(), 2);

}


TEST(tensor, sparse) {

    spvec va(make_shape(1000), { {1, 0}, { 2, 2 }, {10, 0}, {99, 99} });
    
    ASSERT_EQ(std::accumulate(va.nzbegin(), va.nzend(), 0.0), 101);
    ASSERT_EQ(va.sum(), 101);

    spvec_<std::complex<double>> vb;
    vb = va;
    ASSERT_TRUE(vb.shape() == va.shape());
    ASSERT_TRUE(vb[2] == 2.0);
    ASSERT_TRUE(vb[99] == 99.0);
    for (size_t i = 0; i < vb.numel(); i++) {
        if (i != 2 && i != 99) {
            ASSERT_TRUE(vb.at_index_const(i) == 0.0);
        }
    }
    ASSERT_TRUE(vb.sum() == 101.0);

    vecx_<std::complex<double>> vc = vb;
    ASSERT_TRUE(vc.shape() == vb.shape());
    ASSERT_TRUE(vc[2] == 2.0);
    ASSERT_TRUE(vc[99] == 99.0);
    for (size_t i = 0; i < vc.numel(); i++) {
        if (i != 2 && i != 99) {
            ASSERT_TRUE(vc.at_index_const(i) == 0.0);
        }
    }

}


TEST(tensor, eval) {

    matx m(make_shape(100, 100));
    m(0, 0) = 1;
    m(last, last) = 9;

    auto spm = eval<false>(m);

}
