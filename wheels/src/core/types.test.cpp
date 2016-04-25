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

  int && i = 1;
  decltype(auto) j = wheels_forward(i);
  using tt = decltype(i);
  int kk = 0;
  int & k = kk;
  decltype(auto) kd = wheels_forward(k);
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

