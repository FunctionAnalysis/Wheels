#include <gtest/gtest.h>

#include "../../tensor"

using namespace wheels;

TEST(tensor, iota) {
  size_t ns = 0;
  for (auto i : iota(50)) {
    ns += i * i;
    print(i, ' ');
  }
  auto n = iota(50).norm();
  println(n);
  ASSERT_EQ(sqrt(ns), n);

  // reverse
  vecx data(make_shape(100));
  std::default_random_engine rng;
  randomize_fields(data, rng);
  vecx rev_data = data[index_tags::last - iota(100)];
  for (auto i : iota(100)) {
    ASSERT_EQ(data[i], rev_data[index_tags::last - i]);
  }
}