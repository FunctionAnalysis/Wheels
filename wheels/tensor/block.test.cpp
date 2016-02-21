#include "../tensor.hpp"
#include <gtest/gtest.h>

using namespace wheels;
using namespace wheels::index_tags;
using namespace wheels::literals;

TEST(tensor, block) {
  auto a = std::get<0>(meshgrid(make_shape(5, 4)));
  for (auto i : iota(a.size(0_c))) {
    println(a.block(i, everything));
  }
  auto b = a.eval();
  b.block(everything, 0) = b.block(everything, 1) + 2;
  println(b);
}