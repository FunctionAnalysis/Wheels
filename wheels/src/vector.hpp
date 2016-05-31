#pragma once

#include "tensor_base.hpp"
#include "ewise.hpp"
#include "tensor.hpp"

#include "vector_fwd.hpp"

namespace wheels {

// distance
template <class ET1, class ShapeT1, class T1, class ET2, class ShapeT2,
          class T2>
constexpr auto distance(const tensor_base<ET1, ShapeT1, T1> &t1,
                        const tensor_base<ET2, ShapeT2, T2> &t2) {
  return norm_of(t1.derived() - t2.derived());
}

// dot(ts1, ts2);
template <class ET1, class ShapeT1, class T1, class ET2, class ShapeT2,
          class T2>
auto dot(const tensor_base<ET1, ShapeT1, T1> &t1,
         const tensor_base<ET2, ShapeT2, T2> &t2) {
  using result_t = std::common_type_t<ET1, ET2>;
  assert(shape_of(t1.derived()) == shape_of(t2.derived()));
  result_t result = 0.0;
  for_each_element(behavior_flag<unordered>(),
                   [&result](auto &&e1, auto &&e2) { result += e1 * e2; },
                   t1.derived(), t2.derived());
  return result;
}

// auto cross(ts1, ts2);
template <class E1, class ST1, class NT1, class T1, class E2, class ST2,
          class NT2, class T2>
constexpr auto cross(const tensor_base<E1, tensor_shape<ST1, NT1>, T1> &a,
                     const tensor_base<E2, tensor_shape<ST2, NT2>, T2> &b) {
  using result_t = std::common_type_t<E1, E2>;
  return vec_<result_t, 3>(a.y() * b.z() - a.z() * b.y(),
                           a.z() * b.x() - a.x() * b.z(),
                           a.x() * b.y() - a.y() * b.x());
}

// 1 dimensional tensor (vector)
template <class ET, class ST, class NT, class T>
class tensor_base<ET, tensor_shape<ST, NT>, T> : public tensor_core<T> {
public:
  using value_type = ET;
  using shape_type = tensor_shape<ST, NT>;
  static constexpr size_t rank = 1;
  using tensor_type = tensor<value_type, shape_type>;
  static_assert(!is_tensor_shape<ET>::value,
                "value_type should not be a tensor_shape");

  static constexpr auto get_value_type() { return types<value_type>(); }
  static constexpr auto get_shape_type() { return types<shape_type>(); }

  const tensor_base &base() const { return *this; }

  constexpr tensor_type eval() const & { return tensor_type(this->derived()); }
  tensor_type eval() && { return tensor_type(std::move(this->derived())); }
  constexpr operator tensor_type() const { return eval(); }

  // xyzw
  constexpr decltype(auto) x() const {
    return ::wheels::element_at(this->derived(), 0);
  }
  constexpr decltype(auto) y() const {
    return ::wheels::element_at(this->derived(), 1);
  }
  constexpr decltype(auto) z() const {
    return ::wheels::element_at(this->derived(), 2);
  }
  constexpr decltype(auto) w() const {
    return ::wheels::element_at(this->derived(), 3);
  }

  decltype(auto) x() { return ::wheels::element_at(this->derived(), 0); }
  decltype(auto) y() { return ::wheels::element_at(this->derived(), 1); }
  decltype(auto) z() { return ::wheels::element_at(this->derived(), 2); }
  decltype(auto) w() { return ::wheels::element_at(this->derived(), 3); }

  // rgba
  constexpr decltype(auto) r() const {
    return ::wheels::element_at(this->derived(), 0);
  }
  constexpr decltype(auto) g() const {
    return ::wheels::element_at(this->derived(), 1);
  }
  constexpr decltype(auto) b() const {
    return ::wheels::element_at(this->derived(), 2);
  }
  constexpr decltype(auto) a() const {
    return ::wheels::element_at(this->derived(), 3);
  }

  decltype(auto) r() { return ::wheels::element_at(this->derived(), 0); }
  decltype(auto) g() { return ::wheels::element_at(this->derived(), 1); }
  decltype(auto) b() { return ::wheels::element_at(this->derived(), 2); }
  decltype(auto) a() { return ::wheels::element_at(this->derived(), 3); }

  // dot
  template <class ST2, class NT2, class ET2, class T2>
  constexpr decltype(auto)
  dot(const tensor_base<ET2, tensor_shape<ST2, NT2>, T2> &t) const {
    return ::wheels::dot(this->derived(), t);
  }

  // cross
  template <class ST2, class NT2, class ET2, class T2>
  constexpr decltype(auto)
  cross(const tensor_base<ET2, tensor_shape<ST2, NT2>, T2> &t) const {
    return ::wheels::cross(this->derived(), t);
  }
};
}
