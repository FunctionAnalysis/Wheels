#include <gtest/gtest.h>

#include "const_expr.hpp"
#include "object.hpp"
#include "object_fwd.hpp"

using namespace wheels;
using namespace wheels::literals;

template <class T> std::string get_kind_name(const category::object<T> &) {
  return "object";
}
template <class T> std::string get_kind_name(const category::other<T> &) {
  return "other";
}

TEST(core, object) {
  ASSERT_TRUE(get_kind_name(category::identify(1)) == "other");
  ASSERT_TRUE(get_kind_name(category::identify(1_symbol)) == "object");
  int a[5];
  category::identify(a);
}