#include <gtest/gtest.h>

#include "../../tensor"

using namespace wheels;
using namespace wheels::index_tags;
using namespace wheels::literals;

TEST(tensor, block) {
  auto a = meshgrid(make_shape(5, 4), 0_c);
  auto bb = a.block(0, range(0, last));
  for (auto i : iota(a.size(0_c))) {
    println(a.block(i, range(0, last)));
  }
  auto b = a.eval();
  b.block(range(0, last), 0) = b.block(range(0, last), 1) + 2;
  println(b);
  println(b.block(range(0, 2, last), range(0, 3, last)));

  auto xy = coordinate(make_shape(5, 4));
  println(xy);
  println(xy.block(range(0, 2, last), range(last, -1, first)));
}