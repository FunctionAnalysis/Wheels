#include <gtest/gtest.h>

#include <random>

#include "../core/fields.hpp"

#include "constants.hpp"
#include "downgrade.hpp"
#include "tensor.hpp"
#include "upgrade.hpp"

using namespace wheels;
using namespace wheels::literals;

TEST(tensor, subwise) {
  ASSERT_TRUE(downgrade(ones(5ull, 4_sizec, 3_sizec), 1_c).sum() ==
              constants(make_shape(4_c, 3_c), 5));
  ASSERT_TRUE(downgrade(ones(5ull, 4_sizec, 3_sizec), 2_c).sum() ==
              constants(make_shape(3_c), 20));

  matx_<vec3> m(make_shape(50, 50));
  std::default_random_engine rng;
  randomize_fields(m, rng);
  decltype(auto) original = downgrade(upgrade_as_subtensor(m), 1_c);
  ASSERT_TRUE(&original == &m);

  auto c = downgrade(m, 1_c);
  decltype(auto) original2 = upgrade_as_subtensor(c);
  ASSERT_TRUE(&original2 == &m);
}