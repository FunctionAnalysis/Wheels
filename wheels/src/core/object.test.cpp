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
template <class T>
std::string get_kind_name(const category::std_container<T> &) {
  return "std_container";
}

TEST(core, object) {
  ASSERT_TRUE(get_kind_name(category::identify(1)) == "other");
  ASSERT_TRUE(get_kind_name(category::identify(1_arg)) == "object");
  int a[5];
  ASSERT_TRUE(get_kind_name(category::identify(a)) == "std_container");
}