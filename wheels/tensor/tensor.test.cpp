#include <gtest/gtest.h>

#include "tensor.hpp"

using namespace wheels;

TEST(tensor, tensor) {

    tensor<double, 2, 3> t1 = { with_elements,
        1, 2, 3, 4, 5, 6
    };
    auto r1 = sin(t1);
    auto r2 = r1 + t1 * 2.0;
    println(type_of(r2));
    tensor<double, 2, 3> tr = normalize(r2);

    ASSERT_TRUE(tr != t1);
    //decltype(b)::value_type bb;

    auto tt = tuplize(tr);

}