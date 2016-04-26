#include <gtest/gtest.h>

#include "object.hpp"
#include "types.hpp"
#include "overloads.hpp"

using namespace wheels;
using namespace wheels::literals;

template <class T> struct A : category::object<A<T>> {};

namespace wheels {
namespace details {
struct _impl {
  constexpr _impl() {}
  template <class T> const char *operator()(A<T> &&v) const { return "rvalue"; }
  template <class T> const char *operator()(A<T> &v) const { return "lvalue"; }
  template <class T> const char *operator()(const A<T> &v) const {
    return "const lvalue";
  }
  template <class TT> const char *operator()(TT &&) const {
    return "other type";
  }
};
}
template <class OpT, class T> auto overload_as(const OpT &, const A<T> &) {
  return details::_impl();
}

template <class T, class K>
auto overload_as(const binary_op_plus &, const A<T> &,
                 const category::other<K> &) {
  return [](auto &&, auto &&) { return "A<T> + int"; };
}

template <class T, class K>
auto overload_as(const binary_op_plus &, const category::other<K> &,
                 const A<T> &) {
  return [](auto &&, auto &&) { return "int + A<T>"; };
}
}

TEST(core, overloads) {
  A<int> a;
  std::cout << -a << std::endl;
  std::cout << -A<int>() << std::endl;
  int ia = 0;
  auto nia = -ia;

  const category::other<int> &ic = category::identify(1);

  auto pp =
      ::wheels::overload_as(binary_op_plus(), ::wheels::category::identify(a),
                            ::wheels::category::identify(1));

  std::cout << a + 1 << std::endl;
  std::cout << 1 + 1 << std::endl;
  std::cout << 1 + a << std::endl;
}
