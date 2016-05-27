#include <gtest/gtest.h>

#include "storage.hpp"

using namespace wheels;

TEST(tensor, storage) {
  storage<std::string, tensor_shape<size_t, size_t>> st1(make_shape(5), "haha");
  ASSERT_EQ(st1.capacity(), 5);
  st1.reshape(make_shape(3));
  ASSERT_EQ(st1.capacity(), 5);
  st1.reshape(make_shape(10));
  ASSERT_EQ(st1.capacity(), 10);
}