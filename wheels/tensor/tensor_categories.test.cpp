#include <gtest/gtest.h>
#include <type_traits>
#include "tensor_categories.hpp"

using namespace wheels;

namespace ddd {
    struct A {};
    struct B {};
    struct C : A, B {};
    struct D : C {};
}




TEST(math, tensor_categories) {

    std::is_standard_layout<ddd::D>::value;

    constexpr ts_category<tensor_shape<size_t, const_size<2>, const_size<2>>, std::array<double, 4>> m(1, 2, 3);

}