#include <gtest/gtest.h>
#include "tensor.hpp"

using namespace wheels;

TEST(math, test) {

    using namespace wheels::literals;
    using namespace wheels::index_tags;

    auto shape = make_tensor_shape<int>(1_c, 2_c, 3_c);
    tensor<decltype(shape), std::array<double, 6>> ts = {1, 2, 3, 4, 5, 6};
    decltype(auto) e = ts[0];

    


}