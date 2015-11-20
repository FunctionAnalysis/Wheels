#include <gtest/gtest.h>
#include "tensor.hpp"

using namespace wheels;

TEST(math, test) {

    using namespace wheels::literals;
    using namespace wheels::index_tags;

    auto shape = make_tensor_shape<int>(1_c, 2_c, 3_c);
    tensor<decltype(shape), std::array<double, 6>> ts(with_elements, 1, 2, 3, 4, 5, 6);
    decltype(auto) e = ts[0];

    auto shape2 = make_tensor_shape<int>(1, 2, 3);
    tensor<decltype(shape2), std::vector<double>> ts2(shape2, with_elements, 1, 2, 3, 4, 5, 6);
    decltype(auto) e2 = ts2[0];

    tensor<decltype(shape2), std::vector<double>> ts3(with_shape, 1, 2, 3);
    decltype(auto) e3 = ts3[0];

    std::vector<double> v = { 1, 2, 3, 4, 5, 6 };
    tensor<decltype(shape2), concurrency::array_view<double, 3>> ts4(shape2, with_args, v);
    
    decltype(auto) ee = ts4.storage()(1);
    storage_traits<concurrency::array_view<double, 3>>::element(ts4.storage(), 1);

    //decltype(auto) e4 = ts4[0];

}