#include <filesystem>
#include <forward_list>
#include <fstream>

#include <gtest/gtest.h>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/array.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/list.hpp>
#include <cereal/types/vector.hpp>

#include "fields.hpp"

using namespace wheels;

template <class T1, class T2> struct B : comparable<B<T1, T2>> {
  B() : v1(), v2() {}
  B(const T1 &a, const T2 &b) : v1(a), v2(b) {}
  template <class V> auto fields(V &&v) & { return v(v1, v2); }
  template <class V> auto fields(V &&v) const & { return v(v1, v2); }
  template <class V> auto fields(V &&v) && {
    return v(std::move(v1), std::move(v2));
  }
  T1 v1;
  T2 v2;
};

TEST(core, fields) {
  auto bt = tuplize(B<char, int>('1', 1));
  static_assert(type_of(bt).decay() == types<std::tuple<char, int>>(), "");
  ASSERT_TRUE(bt == std::make_tuple('1', 1));

  B<int, long long> b2(2, 2ll);
  auto bt2 = tuplize(b2);
  static_assert(type_of(bt2).decay() == types<std::tuple<int &, long long &>>(),
                "");
  ASSERT_TRUE(bt2 == std::make_tuple(2, 2ll));
  ASSERT_TRUE((B<char, int>('2' - '0', 2) == b2));

  bt2 = std::make_tuple(3, 3);
  ASSERT_EQ(b2.v1, 3);
  ASSERT_EQ(b2.v2, 3);
}

struct C {
  B<char, int> b1;
  B<int, char> b2;
};

namespace wheels {
template <class V> auto fields(const C &c, V &&visitor) {
  return visitor(c.b1, c.b2);
}
template <class V> auto fields(C &c, V &&visitor) {
  return visitor(c.b1, c.b2);
}
}

TEST(core, fields2) {
  C c = {{'1', 1}, {1, '1'}};
  auto ct = tuplize(c);
  static_assert(type_of(ct).decay() ==
                    types<std::tuple<std::tuple<char &, int &>,
                                     std::tuple<int &, char &>>>(),
                "");
  B<int, int> b1 = {0, 0};
  B<int, int> b2 = {0, 0};
  std::forward_as_tuple(tuplize(b1), tuplize(b2)) = tuplize(c);
  ASSERT_EQ(b1.v1, '1');
  ASSERT_EQ(b1.v2, 1);
  ASSERT_EQ(b2.v1, 1);
  ASSERT_EQ(b2.v2, '1');

  ASSERT_TRUE(b1 != b2);
}

struct D {
  std::vector<C> cs;
  B<int, long> b;
  template <class V> auto fields(V &&v) & { return v(cs, b); }
  template <class V> auto fields(V &&v) const & { return v(cs, b); }
};

TEST(core, fields3) {
  D d = {{{{'1', 1}, {1, '1'}}, {{'2', 2}, {2, '2'}}, {{'3', 3}, {3, '3'}}},
         {10, 10}};
  auto dt = tuplize(d);
  std::vector<D> ds = {d, d};
  auto dts = tuplize(ds);
  decltype(auto) dt1 = dts[0];
  C c = {{'4', 4}, {4, '4'}};
  std::get<0>(dt1)[0] = tuplize(c);

  ASSERT_EQ(ds[0].cs[0].b1.v1, '4');

  std::list<D> ds2 = {ds.front(), d};
  ASSERT_TRUE(tuplize(ds) == tuplize(ds2));
  ASSERT_TRUE(tuplize(ds) >= tuplize(ds2));
  ASSERT_TRUE(tuplize(ds) <= tuplize(ds2));
  ASSERT_FALSE(tuplize(ds) > tuplize(ds2));
  ASSERT_FALSE(tuplize(ds) < tuplize(ds2));
}

struct E {};

TEST(core, fields4) {

  auto et = tuplize(E());
  auto ets = tuplize(std::make_tuple(E(), 1, std::vector<E>(10)));
  ASSERT_TRUE(ets == ets);
  ASSERT_TRUE(ets <= ets);
}

struct F_tag {};

template <class T> class F : public convertible<F<T>, F_tag> {
public:
  constexpr F() : val() {}
  constexpr F(T v) : val(v) {}
  T val;

  using convertible<F<T>, F_tag>::operator=;

  template <class V> auto fields(V &&v) & { return v(val); }
  template <class V> auto fields(V &&v) const & { return v(val); }
};

TEST(core, fields5) {

  F<float> f1 = 1;
  F<double> f2 = 2;
  ASSERT_TRUE(tuplize(f1) != tuplize(f2));
  f2 = f1;
  ASSERT_TRUE(tuplize(f1) == tuplize(f2));
}

template <class T> struct dummy {};

struct X {
  bool a, b, c;
  int d;
  X() : a(true), b(true), c(false), d(100) {}
  template <class V> auto fields(V &&v) & { return v(a, b, c, d); }
  template <class V> auto fields(V &&v) const & { return v(a, b, c, d); }
};

struct Y {
  std::pair<X, char> d;
  Y() : d(X(), 'H') {}
  template <class V> auto fields(V &&v) & { return v(d); }
  template <class V> auto fields(V &&v) const & { return v(d); }
};

struct Z {
  Y ys[3];
  X x;
  Z() {}
  template <class V> auto fields(V &&v) & { return v(ys, x); }
  template <class V> auto fields(V &&v) const & { return v(ys, x); }
};

struct KK {
  auto const_call() const { return yes(); }
  auto const_call() { return no(); }
};

struct PP {
  KK &kk;
  auto call() const { return kk.const_call(); }
};

TEST(core, fields6) {
  std::vector<Z> zs(2);
  traverse_fields(zs, [](auto &e) { e = 0; });
  ASSERT_TRUE(!any_of_fields(zs, [](auto &&e) { return e != 0; }));
  ASSERT_TRUE(all_of_fields(zs, [](auto &&e) { return e == 0; }));
  traverse_fields(zs, [](auto &e) { e = 1; });
  ASSERT_TRUE(!any_of_fields(zs, [](auto &&e) { return e != 1; }));
  ASSERT_TRUE(all_of_fields(zs, [](auto &&e) { return e == 1; }));
}

TEST(core, fields7) {
  int data[2][3][4];
  std::default_random_engine rng;
  randomize_fields(data, rng);
  ASSERT_TRUE(tuplize(data) == tuplize(data));
}
