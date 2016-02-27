#include <gtest/gtest.h>

#include "../../tensor"

using namespace wheels;
using namespace wheels::literals;
using namespace wheels::index_tags;

TEST(tensor, constants) {
  for (auto i : ones(5, 6, 7) * 5) {
    ASSERT_TRUE(i == 5);
  }
}