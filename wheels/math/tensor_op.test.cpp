#include <gtest/gtest.h>

#include "tensor_op.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;

TEST(math, tensor_constants) {
    constexpr auto a = zeros(make_shape(1_c, 2_c, 3_c));
    auto k = a(0, 1, 2);
}
