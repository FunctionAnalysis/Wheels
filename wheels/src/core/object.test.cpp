#include <gtest/gtest.h>

#include "../../core"

using namespace wheels;
using namespace wheels::literals;

template <class T> std::string get_kind_name(const kinds::object<T> &) {
  return "object";
}
template <class T> std::string get_kind_name(const kinds::other<T> &) {
  return "other";
}

TEST(core, object) {
  ASSERT_TRUE(get_kind_name(kinds::identify(1)) == "other");
  ASSERT_TRUE(get_kind_name(kinds::identify(1_symbol)) == "object");
}