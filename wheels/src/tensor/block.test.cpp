#include <gtest/gtest.h>

#include "../../tensor"

using namespace wheels;
using namespace wheels::index_tags;
using namespace wheels::literals;

TEST(tensor, block) {
  auto a = meshgrid_at(make_shape(5, 4), 0_c);
  for (auto i : iota(a.size(0_c))) {
    println(a.block(i, everything));
  }
  auto b = a.eval();
  b.block(everything, 0) = b.block(everything, 1) + 2;
  println(b);
}