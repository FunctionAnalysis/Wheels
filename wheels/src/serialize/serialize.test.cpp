#include <gtest/gtest.h>

#include "../../core"
#include "../../tensor"

#include "serialize.hpp"

using namespace wheels;
using namespace wheels::literals;

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

#if defined(wheels_compiler_msc)
TEST(serialize, basic) {
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

TEST(serialize, tensor) {
  write_tmp("vec3.cereal", vec3(1, 2, 3));
  vec3 v;
  read_tmp("vec3.cereal", v);
  ASSERT_TRUE(v == vec3(1, 2, 3));
  vecx v2;
  read_tmp("vec3.cereal", v2);
  ASSERT_TRUE(v2 == vec3(1, 2, 3));

  auto vb = ones<bool>(5).eval();
  write_tmp("vecb5", vb);
  vecx_<bool> vb2;
  read_tmp("vecb5", vb2);
  ASSERT_TRUE(vb == vb2);
}

TEST(serialize, shape) {
  auto inshape = make_shape(2_c, 3_c, 4_c);
  write_tmp("shape1.cereal", inshape);
  shape_of_rank<int, 3> outshape;
  read_tmp("shape1.cereal", outshape);
  ASSERT_TRUE(inshape == outshape);
  auto inshape2 = make_shape(5_c, 4, 3, 2_c, 1);
  write_tmp("shape2.cereal", inshape2);
  tensor_shape<int, int, const_int<4>, const_int<3>, int, int> outshape2;
  read_tmp("shape2.cereal", outshape2);
  ASSERT_TRUE(inshape2 == outshape2);
}

#endif