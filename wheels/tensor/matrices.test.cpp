#include <gtest/gtest.h>

#include "matrices.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, matrix_mul) {

    matx a(make_shape(10, 20));
    std::generate(a.begin(), a.end(), std::rand);

    tensor_traits::nonconst_iterator_type<matx>::type t;

    matx b = eye(20, 20);
    matx ab = a * b;

    ASSERT_TRUE(ab == a);
    ASSERT_TRUE(ab * eye(20, 20) == a);

}

TEST(tensor, matrix_vector_mul) {

    vecx v(1, 2, 3, 4, 5, 6, 7, 8);
    matx e = - eye(8, 8);
    ASSERT_TRUE(e * v == -v);

}