#include <gtest/gtest.h>

#include "const_expr.hpp"
#include "what.hpp"

using namespace wheels;
using namespace wheels::literals;

template <class T> std::string get_kind_name(const object_base<T> &) {
  return "object_base";
}
template <class T> std::string get_kind_name(const proxy_base<T> &) {
  return "proxy_base";
}

TEST(core, object) {
  ASSERT_TRUE(get_kind_name(what(1)) == "proxy_base");
  ASSERT_TRUE(get_kind_name(what(1_arg)) == "object_base");
}