#include <gtest/gtest.h>

#include "aligned.hpp"
#include "ewise_ops.hpp"
#include "tensor.hpp"

using namespace wheels;

struct A {};
struct B : A {};
struct C : B {};
struct D : C {};
struct E : D {};

A foo(const A &) { return A(); }
C foo(const C &) { return C(); }

TEST(tensor, fun_match_test) {
  using t = decltype(foo(E()));
  static_assert(types<t>() == types<C>(), "");
}

TEST(tensor, aligned_data) {
  vec3 v1(1, 2, 3);
  v1[0];
  vec_<bool, 5> v2;
  v2.ptr();

  element_at_index(v1, 0);

  for_each_element(behavior_flag<index_ascending>(),
                   [](auto e) { std::cout << e << ' '; }, v1);
  auto test = [](auto e) { std::cout << e << '\n'; };
  for_each_element(behavior_flag<unordered>(), test, v1);

  vec_<std::string, 5> vstr;
  vecx_<std::string> vstr2(make_shape(5));
  ASSERT_TRUE(vstr == vstr2);
}