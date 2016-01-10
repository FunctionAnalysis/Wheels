#include <gtest/gtest.h>

#include "serialize.hpp"

using namespace wheels;

struct A : serializable<A> {
  int val;
  A() : val(0) {}
  A(int v) : val(v) {}
  template <class V> auto fields(V &&v) { return v(val); }
};
struct B {};
struct E : serializable<E>, comparable<E> {
  std::vector<A> as;
  std::vector<B> bs;
  E() {}
  E(const std::vector<A> &as) : as(as), bs(as.size()) {}
  template <class V> auto fields(V &&v) { return v(as, bs); }
};

TEST(core, serialize) {
  std::error_code errc;
  auto path = filesystem::temp_directory_path(errc);
  if (!errc) {
    E e({{1}, {2}, {3}, {4}, {5}, {6}});
    auto filepath = path.append("wheels.core.serialize.cereal");
    {
      std::ofstream ofs(filepath);
      cereal::PortableBinaryOutputArchive arc(ofs);
      arc(e);
    }
    E e2;
    ASSERT_FALSE(e == e2);
    {
      std::ifstream ifs(filepath);
      cereal::PortableBinaryInputArchive arc(ifs);
      arc(e2);
    }
    ASSERT_TRUE(e == e2);
  }
}