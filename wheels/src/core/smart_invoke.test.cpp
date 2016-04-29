#include <gtest/gtest.h>

#include "smart_invoke.hpp"

using namespace wheels;
using namespace wheels::literals;

struct tuple_maker {
  template <class... Ts> auto operator()(Ts &&... ts) const {
    return std::make_tuple(ts...);
  }
};

TEST(core, smart_invoke) {
  ASSERT_TRUE(smart_invoke(tuple_maker(), 1, 2) == std::make_tuple(1, 2));
  auto functor = smart_invoke(tuple_maker(), 1_symbol, 2, 0_symbol);
  ASSERT_TRUE(functor(std::string("hahaha"), true) ==
              std::make_tuple(true, 2, std::string("hahaha")));
}