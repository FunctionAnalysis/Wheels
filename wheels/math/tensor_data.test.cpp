#include <gtest/gtest.h>

#include "tensor_data.hpp"

using namespace wheels;

TEST(math, test) {

    using namespace wheels::literals;
    
    auto shape = make_shape<int>(1, 2_c, 3, 4_c);

    static_assert(is_constructible_with_shape<std::array<double, 5>>::value, "");
    static_assert(is_constructible_with_elements<std::array<double, 5>>::value, "");
    static_assert(is_constructible_with_shape_elements<std::array<double, 5>>::value, "");

    static_assert(is_element_accessible_at_index<platform_cpu, std::array<double, 5>>::value, "");
    static_assert(!is_element_accessible_at_index<platform_amp, std::array<double, 5>>::value, "");    
    static_assert(!is_element_accessible_at_subs<platform_cpu, std::array<double, 5>>::value, "");
    static_assert(!is_element_accessible_at_subs<platform_amp, std::array<double, 5>>::value, "");


    static_assert(is_constructible_with_shape<concurrency::array_view<double, 3>>::value, "");
    static_assert(!is_constructible_with_elements<concurrency::array_view<double, 3>>::value, "");
    static_assert(!is_constructible_with_shape_elements<concurrency::array_view<double, 3>>::value, "");

    static_assert(!is_element_accessible_at_index<platform_cpu, concurrency::array_view<double, 3>>::value, "");
    static_assert(!is_element_accessible_at_index<platform_amp, concurrency::array_view<double, 3>>::value, "");
    static_assert( is_element_accessible_at_subs<platform_cpu,  concurrency::array_view<double, 3>>::value, "");
    static_assert( is_element_accessible_at_subs<platform_amp,  concurrency::array_view<double, 3>>::value, "");



}