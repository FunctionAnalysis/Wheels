#include <gtest/gtest.h>

#include "map.hpp"
#include "reshape.hpp"
#include "shape.hpp"
#include "tensor.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, reshape) {
  auto s1 = "hello world!"_ts;
  auto s1e = reshape(s1, make_shape(3, 4));
  println(s1e);
  auto s1ee = reshape(s1e, make_shape(2, 6));
  println(s1ee);
  auto s1eee = reshape(s1e, make_shape(1, 12));
  println(s1eee);
}