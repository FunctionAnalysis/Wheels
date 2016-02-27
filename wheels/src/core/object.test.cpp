#include <gtest/gtest.h>

#include "../../core"

using namespace wheels;
using namespace wheels::literals;

TEST(core, object) {
  ASSERT_TRUE(type_of(identify(1)) == types<const other<int> &>());
  ASSERT_TRUE(type_of(identify(1_symbol)) == types<const const_symbol<1> &>());
}