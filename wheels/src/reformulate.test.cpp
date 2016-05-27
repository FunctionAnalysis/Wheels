#include <gtest/gtest.h>

#include "cat.hpp"
#include "constants.hpp"
#include "downgrade.hpp"
#include "ewise.hpp"
#include "permute.hpp"
#include "reformulate.hpp"
#include "tensor.hpp"
#include "upgrade.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, reformulate) {
  std::default_random_engine rng;
  auto a = rand(make_shape(10, 20, 30), rng);
  auto b1 = reformulate(a, make_shape(30, 20, 10), [](auto i, auto j, auto k) {
    return std::make_tuple(k, j, i);
  });
  auto b2 = permute(a, 2_c, 1_c, 0_c);
  ASSERT_TRUE(b1 == b2);

  auto c1 = reformulate(a, make_shape(60, 30), [](auto i, auto j) {
    return std::make_tuple(0, i % 20, j);
  });
  auto c2 =
      cat(subtensor_at(a, 0), subtensor_at(a, 0), subtensor_at(a, 0)).eval();

  ASSERT_TRUE(c1 == c2);
}

TEST(tensor, reformulate2) {
  std::default_random_engine rng;
  auto a = rand(make_shape(10, 20, 30), rng);
  auto c1 = reformulate(a, make_shape(60, 30), [](auto i, auto j) {
    return std::make_tuple(0, i % 20, j);
  });

  auto c3 = repeat(subtensor_at(a, 0), 3, 1);
  ASSERT_TRUE(c1 == c3);
}

TEST(tensor, downgrade_upgrade) {
  ASSERT_TRUE(ones(5ull, 4_sizec, 3_sizec).downgraded(1_c).sum() ==
              constants(make_shape(4_c, 3_c), 5));
  ASSERT_TRUE(ones(5ull, 4_sizec, 3_sizec).downgraded(2_c).sum() ==
              constants(make_shape(3_c), 20));

  matx_<vec3> m(make_shape(50, 50));
  std::default_random_engine rng;
  randomize(m, rng);
  decltype(auto) original = m.upgraded_all().downgraded(1_c);
  ASSERT_TRUE(&original == &m);

  auto c = m.downgraded(1_c);
  decltype(auto) original2 = c.upgraded_all();
  ASSERT_TRUE(&original2 == &m);
}