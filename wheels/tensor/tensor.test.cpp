#include <gtest/gtest.h>

#include "tensor.hpp"

using namespace wheels;

TEST(tensor, tensor) {

    static_assert(std::is_standard_layout<mat2>::value, "");

    mat_<double, 2, 3> t1(
        1, 2, 3, 4, 5, 6
    );
    auto r1 = sin(t1);
    auto r2 = r1 + t1 * 2.0;
    println(type_of(r2));
    mat_<double, 3, 2> tr = normalize(r2).t();

    auto k = t1 * tr;
    //auto kk = cube2() * cube2();
    cube2 c;
    auto kk = ewise_mul(cube2(), cube2()).eval();


    ASSERT_TRUE(tr.t() != t1);
    //decltype(b)::value_type bb;

    auto tt = tuplize(tr);

    vec3 v1(1, 2, 3);

    vecx vs(1, 2, 3, 4, 5);
    ASSERT_TRUE(vs.numel() == 5);
    tensor<tensor_shape<size_t, const_size<2>, size_t, const_size<2>>, double> ts(1, 2, 3, 4, 5, 6, 7, 8);
    ASSERT_EQ(ts.numel(), 8);

}