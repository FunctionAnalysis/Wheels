#include <gtest/gtest.h>
#include "type.hpp"

using namespace wheels;

TEST(core, type) {

    int a = 0, b = 0;
    const auto ta = type_of(a);
    const auto tb = type_of(b);
    auto tbd = tb.decay();
    static_assert(decltype(ta == tb)::value, "");

}