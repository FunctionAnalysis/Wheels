#include <gtest/gtest.h>

#include "geometry.hpp"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;

TEST(math, matrix) {
    
    auto v = normalize(ones(3));
    auto m1 = make_rotate3(vec3(ones(3)), 5.0);

}
