#include <gtest/gtest.h>

#include "remap.hpp"

using namespace wheels;

TEST(tensor, remap) {

  vecx anchors(1, 2, 3);
  auto remapped = remap(
      anchors, make_shape(12),
      [](size_t s) { return std::vector<double>{(s + 1) / 4.0 - 1}; }, -1);
  auto result = remapped.eval();
}