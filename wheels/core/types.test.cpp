#include "types.hpp"
#include <gtest/gtest.h>

using namespace wheels;

TEST(core, type) {
  using namespace wheels::literals;

  int a = 0, b = 0;
  double c = 0;
  std::string d = "";
  auto t = type_of(a, b, c, d);

  static_assert(t[0_c].is<int &>() && t[1_c].is<int &>() &&
                    t[2_c].is<double &>() && t[3_c].is<std::string &>(),
                "");
}

template <class T> struct identity {using type = T;};

template <class T, class BaseT,
          bool IsRValue = std::is_rvalue_reference<T>::value,
          bool ValidBase = std::is_base_of<BaseT, std::decay_t<T>>::value>
struct divided {};
template <class T, class BaseT> struct divided<T, BaseT, true, true> {
  using type = T;
};
template <class T, class BaseT> struct divided<T, BaseT, false, true> {
  using type = T;
};

template <class T, class BaseT>
using divided_t = typename divided<T, BaseT>::type;

struct A {};
struct B : A {};
struct C {};
struct D : C {};

template <class T> int foo(typename identity<T>::type & t) {}
//template <class T> A foo(divided_t<T, A> a) {}
//template <class T> C foo(divided_t<T, C> c) {}

