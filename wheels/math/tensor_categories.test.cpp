#include <gtest/gtest.h>
#include "tensor_categories.hpp"

using namespace wheels;

TEST(math, tensor_categories) {

    constexpr ts_category<tensor_shape<size_t, const_size<2>, const_size<2>>, std::array<double, 4>> m(1, 2, 3);
    constexpr auto e1 = m.at_index_const(0);
    auto ee1 = m.at_subs_const(0, 0);
    static_assert(e1 == 1, "");
    

}