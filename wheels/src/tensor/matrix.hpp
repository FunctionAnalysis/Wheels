#pragma once

#include "base.hpp"
#include "block.hpp"
#include "constants.hpp"
#include "ewise.hpp"
#include "iota.hpp"
#include "tensor.hpp"

#include "matrix_fwd.hpp"

namespace wheels {

// matrix base
template <class T> struct matrix_base : tensor_core<T> {
  constexpr auto rows() const { return this->size(const_index<0>()); }
  constexpr auto cols() const { return this->size(const_index<1>()); }

  constexpr decltype(auto) t() const & { return transpose(this->derived()); }
  decltype(auto) t() & { return transpose(this->derived()); }
  decltype(auto) t() && { return transpose(std::move(this->derived())); }

  constexpr decltype(auto) row(size_t r) const & {
    return at_block(this->derived(), constants(make_shape(), std::move(r)),
                    iota<size_t>(cols()));
  }
  decltype(auto) row(size_t r) & {
    return at_block(this->derived(), constants(make_shape(), std::move(r)),
                    iota<size_t>(cols()));
  }
  decltype(auto) row(size_t r) && {
    return at_block(std::move(this->derived()),
                    constants(make_shape(), std::move(r)),
                    iota<size_t>(cols()));
  }

  constexpr decltype(auto) col(size_t r) const & {
    return at_block(this->derived(), iota<size_t>(rows()),
                    constants(make_shape(), std::move(r)));
  }
  decltype(auto) col(size_t r) & {
    return at_block(this->derived(), iota<size_t>(rows()),
                    constants(make_shape(), std::move(r)));
  }
  decltype(auto) col(size_t r) && {
    return at_block(std::move(this->derived()), iota<size_t>(rows()),
                    constants(make_shape(), std::move(r)));
  }
};

// 2 dimensional tensor (matrix)
template <class ET, class ST, class MT, class NT, class T>
struct tensor_base<ET, tensor_shape<ST, MT, NT>, T> : matrix_base<T> {
  using value_type = ET;
  using shape_type = tensor_shape<ST, MT, NT>;
  static constexpr size_t rank = 2;
  using tensor_type = tensor<value_type, shape_type>;
  static_assert(!is_tensor_shape<ET>::value,
                "value_type should not be a tensor_shape");

  static constexpr auto get_value_type() { return types<value_type>(); }
  static constexpr auto get_shape_type() { return types<shape_type>(); }

  const tensor_base &base() const { return *this; }

  constexpr tensor_type eval() const & { return tensor_type(this->derived()); }
  tensor_type eval() && { return tensor_type(std::move(this->derived())); }
  constexpr operator tensor_type() const { return eval(); }
};

// col vec
template <class ET, class ST, class MT, class T>
struct tensor_base<ET, tensor_shape<ST, MT, const_ints<ST, (ST)1>>, T>
    : matrix_base<T> {
  using value_type = ET;
  using shape_type = tensor_shape<ST, MT, const_ints<ST, (ST)1>>;
  static constexpr size_t rank = 2;
  using tensor_type = tensor<value_type, shape_type>;
  static_assert(!is_tensor_shape<ET>::value,
                "value_type should not be a tensor_shape");

  static constexpr auto get_value_type() { return types<value_type>(); }
  static constexpr auto get_shape_type() { return types<shape_type>(); }

  const tensor_base &base() const { return *this; }

  constexpr tensor_type eval() const & { return tensor_type(this->derived()); }
  tensor_type eval() && { return tensor_type(std::move(this->derived())); }
  constexpr operator tensor_type() const { return this->eval(); }

  constexpr decltype(auto) to_vec() const & {
    return ::wheels::reshape(this->derived(), make_shape(this->rows()));
  }
  decltype(auto) to_vec() & {
    return ::wheels::reshape(this->derived(), make_shape(this->rows()));
  }
  decltype(auto) to_vec() && {
    return ::wheels::reshape(std::move(this->derived()),
                             make_shape(this->rows()));
  }

  // xyzw
  constexpr decltype(auto) x() const {
    return ::wheels::element_at(this->derived(), 0, 0);
  }
  constexpr decltype(auto) y() const {
    return ::wheels::element_at(this->derived(), 1, 0);
  }
  constexpr decltype(auto) z() const {
    return ::wheels::element_at(this->derived(), 2, 0);
  }
  constexpr decltype(auto) w() const {
    return ::wheels::element_at(this->derived(), 3, 0);
  }

  decltype(auto) x() { return ::wheels::element_at(this->derived(), 0, 0); }
  decltype(auto) y() { return ::wheels::element_at(this->derived(), 1, 0); }
  decltype(auto) z() { return ::wheels::element_at(this->derived(), 2, 0); }
  decltype(auto) w() { return ::wheels::element_at(this->derived(), 3, 0); }

  // rgba
  constexpr decltype(auto) r() const {
    return ::wheels::element_at(this->derived(), 0, 0);
  }
  constexpr decltype(auto) g() const {
    return ::wheels::element_at(this->derived(), 1, 0);
  }
  constexpr decltype(auto) b() const {
    return ::wheels::element_at(this->derived(), 2, 0);
  }
  constexpr decltype(auto) a() const {
    return ::wheels::element_at(this->derived(), 3, 0);
  }

  decltype(auto) r() { return ::wheels::element_at(this->derived(), 0, 0); }
  decltype(auto) g() { return ::wheels::element_at(this->derived(), 1, 0); }
  decltype(auto) b() { return ::wheels::element_at(this->derived(), 2, 0); }
  decltype(auto) a() { return ::wheels::element_at(this->derived(), 3, 0); }
};

// row vec
template <class ET, class ST, class NT, class T>
struct tensor_base<ET, tensor_shape<ST, const_ints<ST, (ST)1>, NT>, T>
    : matrix_base<T> {
  using value_type = ET;
  using shape_type = tensor_shape<ST, const_ints<ST, (ST)1>, NT>;
  static constexpr size_t rank = 2;
  using tensor_type = tensor<value_type, shape_type>;
  static_assert(!is_tensor_shape<ET>::value,
                "value_type should not be a tensor_shape");

  static constexpr auto get_value_type() { return types<value_type>(); }
  static constexpr auto get_shape_type() { return types<shape_type>(); }

  const tensor_base &base() const { return *this; }

  constexpr tensor_type eval() const & { return tensor_type(this->derived()); }
  tensor_type eval() && { return tensor_type(std::move(this->derived())); }
  constexpr operator tensor_type() const { return this->eval(); }

  constexpr decltype(auto) to_vec() const & {
    return ::wheels::reshape(this->derived(), make_shape(this->cols()));
  }
  decltype(auto) to_vec() & {
    return ::wheels::reshape(this->derived(), make_shape(this->cols()));
  }
  decltype(auto) to_vec() && {
    return ::wheels::reshape(std::move(this->derived()),
                             make_shape(this->cols()));
  }

  // xyzw
  constexpr decltype(auto) x() const {
    return ::wheels::element_at(this->derived(), 0, 0);
  }
  constexpr decltype(auto) y() const {
    return ::wheels::element_at(this->derived(), 0, 1);
  }
  constexpr decltype(auto) z() const {
    return ::wheels::element_at(this->derived(), 0, 2);
  }
  constexpr decltype(auto) w() const {
    return ::wheels::element_at(this->derived(), 0, 3);
  }

  decltype(auto) x() { return ::wheels::element_at(this->derived(), 0, 0); }
  decltype(auto) y() { return ::wheels::element_at(this->derived(), 0, 1); }
  decltype(auto) z() { return ::wheels::element_at(this->derived(), 0, 2); }
  decltype(auto) w() { return ::wheels::element_at(this->derived(), 0, 3); }

  // rgba
  constexpr decltype(auto) r() const {
    return ::wheels::element_at(this->derived(), 0, 0);
  }
  constexpr decltype(auto) g() const {
    return ::wheels::element_at(this->derived(), 0, 1);
  }
  constexpr decltype(auto) b() const {
    return ::wheels::element_at(this->derived(), 0, 2);
  }
  constexpr decltype(auto) a() const {
    return ::wheels::element_at(this->derived(), 0, 3);
  }

  decltype(auto) r() { return ::wheels::element_at(this->derived(), 0, 0); }
  decltype(auto) g() { return ::wheels::element_at(this->derived(), 0, 1); }
  decltype(auto) b() { return ::wheels::element_at(this->derived(), 0, 2); }
  decltype(auto) a() { return ::wheels::element_at(this->derived(), 0, 3); }
};

// matrix * matrix -> matrix
template <class EleT, class ShapeT, class A, class B>
class matrix_mul_result<EleT, ShapeT, A, B, true, true>
    : public tensor_base<EleT, ShapeT,
                         matrix_mul_result<EleT, ShapeT, A, B, true, true>> {
public:
  using value_type = EleT;
  using shape_type = ShapeT;
  constexpr matrix_mul_result(A &&aa, B &&bb)
      : _a(std::forward<A>(aa)), _b(std::forward<B>(bb)) {}
  constexpr auto shape() const {
    return make_shape(size_at(_a, const_index<0>()),
                      size_at(_b, const_index<1>()));
  }
  template <class SubT1, class SubT2>
  decltype(auto) at_subs(const SubT1 &s1, const SubT2 &s2) const {
    EleT result = types<EleT>::zero();
    for (size_t i = 0; i < size_at(_a, const_index<1>()); i++) {
      result += element_at(_a, s1, i) * element_at(_b, i, s2);
    }
    return result;
  }

private:
  A _a;
  B _b;
};

// matrix * vector -> vector
template <class EleT, class ShapeT, class A, class B>
class matrix_mul_result<EleT, ShapeT, A, B, true, false>
    : public tensor_base<EleT, ShapeT,
                         matrix_mul_result<EleT, ShapeT, A, B, true, false>> {
public:
  using value_type = EleT;
  using shape_type = ShapeT;
  constexpr matrix_mul_result(A &&aa, B &&bb)
      : _a(std::forward<A>(aa)), _b(std::forward<B>(bb)) {}
  constexpr auto shape() const {
    return make_shape(size_at(_a, const_index<0>()));
  }
  template <class SubT> decltype(auto) at_subs(const SubT &s) const {
    EleT result = types<EleT>::zero();
    for (size_t i = 0; i < size_at(_a, const_index<1>()); i++) {
      result += element_at(_a, s, i) * element_at(_b, i);
    }
    return result;
  }

private:
  A _a;
  B _b;
};

// vector * matrix -> vector
template <class EleT, class ShapeT, class A, class B>
class matrix_mul_result<EleT, ShapeT, A, B, false, true>
    : public tensor_base<EleT, ShapeT,
                         matrix_mul_result<EleT, ShapeT, A, B, false, true>> {
public:
  using value_type = EleT;
  using shape_type = ShapeT;
  constexpr matrix_mul_result(A &&aa, B &&bb)
      : _a(std::forward<A>(aa)), _b(std::forward<B>(bb)) {}
  constexpr auto shape() const {
    return make_shape(size_at(_b, const_index<1>()));
  }
  template <class SubT> decltype(auto) at_subs(const SubT &s) const {
    EleT result = types<EleT>::zero();
    for (size_t i = 0; i < size_at(_a, const_index<0>()); i++) {
      result += element_at(_a, i) * element_at(_b, i, s);
    }
    return result;
  }

private:
  A _a;
  B _b;
};

template <class EleT, class ShapeT, bool AIsMat, bool BIsMat, class A, class B>
constexpr auto make_matrix_mul_result(A &&a, B &&b) {
  return matrix_mul_result<EleT, ShapeT, A, B, AIsMat, BIsMat>(
      std::forward<A>(a), std::forward<B>(b));
}

// shape_of
template <class EleT, class ShapeT, class A, class B, bool AIsMat, bool BIsMat>
constexpr auto
shape_of(const matrix_mul_result<EleT, ShapeT, A, B, AIsMat, BIsMat> &m) {
  return m.shape();
}
// element_at
template <class EleT, class ShapeT, class A, class B, bool AIsMat, bool BIsMat,
          class... SubTs>
constexpr decltype(auto)
element_at(const matrix_mul_result<EleT, ShapeT, A, B, AIsMat, BIsMat> &m,
           const SubTs &... subs) {
  return m.at_subs(subs...);
}

template <class ST1, class MT1, class NT1, class E1, class T1, class ST2,
          class MT2, class NT2, class E2, class T2>
auto overload_as(const func_base<binary_op_mul> &,
                 const tensor_base<E1, tensor_shape<ST1, MT1, NT1>, T1> &,
                 const tensor_base<E2, tensor_shape<ST2, MT2, NT2>, T2> &) {
  return [](auto &&a, auto &&b) {
    assert(size_at(a, const_index<1>()) == size_at(b, const_index<0>()));
    using shape_t = std::decay_t<decltype(make_shape(
        size_at(a, const_index<0>()), size_at(b, const_index<1>())))>;
    return make_matrix_mul_result<std::common_type_t<E1, E2>, shape_t, true,
                                  true>(wheels_forward(a), wheels_forward(b));
  };
}

template <class ST1, class MT1, class NT1, class E1, class T1, class ST2,
          class MT2, class E2, class T2>
auto overload_as(const func_base<binary_op_mul> &,
                 const tensor_base<E1, tensor_shape<ST1, MT1, NT1>, T1> &,
                 const tensor_base<E2, tensor_shape<ST2, MT2>, T2> &) {
  return [](auto &&a, auto &&b) {
    assert(size_at(a, const_index<1>()) == size_at(b, const_index<0>()));
    using shape_t = tensor_shape<ST1, MT1>;
    return make_matrix_mul_result<std::common_type_t<E1, E2>, shape_t, true,
                                  false>(wheels_forward(a), wheels_forward(b));
  };
}

template <class ST1, class MT1, class E1, class T1, class ST2, class MT2,
          class NT2, class E2, class T2>
auto overload_as(const func_base<binary_op_mul> &,
                 const tensor_base<E1, tensor_shape<ST1, MT1>, T1> &,
                 const tensor_base<E2, tensor_shape<ST2, MT2, NT2>, T2> &) {
  return [](auto &&a, auto &&b) {
    assert(size_at(a, const_index<0>()) == size_at(b, const_index<0>()));
    using shape_t = tensor_shape<ST2, NT2>;
    return make_matrix_mul_result<std::common_type_t<E1, E2>, shape_t, false,
                                  true>(wheels_forward(a), wheels_forward(b));
  };
}

// translate (TODO transpose or not?)
template <class E1, class ST1, class MT1, class NT1, class T1, class E2,
          class ST2, class MT2, class T2>
inline auto translate(const tensor_base<E1, tensor_shape<ST1, MT1, NT1>, T1> &m,
                      const tensor_base<E2, tensor_shape<ST2, MT2>, T2> &v) {
  assert(m.rows() == 4 && m.cols() == 4 && v.numel() == 3);
  using ele_t = std::common_type_t<E1, E2>;
  mat_<ele_t, 4, 4> result(m);
  result.row(3) =
      m.row(0) * v[0] + m.row(1) * v[1] + m.row(2) * v[2] + m.row(3);
  return result;
}

// rotate (TODO transpose or not?)
template <class E1, class ST1, class MT1, class NT1, class T1, class E2,
          class EAngle, class ST2, class MT2, class T2>
inline auto rotate(const tensor_base<E1, tensor_shape<ST1, MT1, NT1>, T1> &m,
                   const EAngle &angle,
                   const tensor_base<E2, tensor_shape<ST2, MT2>, T2> &v) {
  assert(m.rows() == 4 && m.cols() == 4 && v.numel() == 3);
  using ele_t = std::common_type_t<E1, EAngle, E2>;
  ele_t const a = angle;
  ele_t const c = std::cos(a);
  ele_t const s = std::sin(a);

  vec_<ele_t, 3> axis(v.derived() / v.norm());
  vec_<ele_t, 3> temp((ele_t(1) - c) * axis);

  mat_<ele_t, 4, 4> rot(m);
  rot(0, 0) = c + temp[0] * axis[0];
  rot(0, 1) = 0 + temp[0] * axis[1] + s * axis[2];
  rot(0, 2) = 0 + temp[0] * axis[2] - s * axis[1];

  rot(1, 0) = 0 + temp[1] * axis[0] - s * axis[2];
  rot(1, 1) = c + temp[1] * axis[1];
  rot(1, 2) = 0 + temp[1] * axis[2] + s * axis[0];

  rot(2, 0) = 0 + temp[2] * axis[0] + s * axis[1];
  rot(2, 1) = 0 + temp[2] * axis[1] - s * axis[0];
  rot(2, 2) = c + temp[2] * axis[2];

  mat_<ele_t, 4, 4> result(m);
  result.row(0) =
      m.row(0) * rot(0, 0) + m.row(1) * rot(0, 1) + m.row(2) * rot(0, 2);
  result.row(1) =
      m.row(0) * rot(1, 0) + m.row(1) * rot(1, 1) + m.row(2) * rot(1, 2);
  result.row(2) =
      m.row(0) * rot(2, 0) + m.row(1) * rot(2, 1) + m.row(2) * rot(2, 2);
  result.row(3) = m.row(3);

  return result;
}

// scale (TODO transpose or not?)
template <class E1, class ST1, class MT1, class NT1, class T1, class E2,
          class ST2, class MT2, class T2>
inline auto scale(const tensor_base<E1, tensor_shape<ST1, MT1, NT1>, T1> &m,
                  const tensor_base<E2, tensor_shape<ST2, MT2>, T2> &v) {
  assert(m.rows() == 4 && m.cols() == 4 && v.numel() == 3);
  using ele_t = std::common_type_t<E1, E2>;
  mat_<ele_t, 4, 4> result(m);
  result.row(0) = m.row(0) * v[0];
  result.row(1) = m.row(1) * v[1];
  result.row(2) = m.row(2) * v[2];
  return result;
}

// camera ops (TODO...)
template <class E1, class ST1, class MT1, class T1, class E2, class ST2,
          class MT2, class T2, class E3, class ST3, class MT3, class T3>
inline auto
look_at_rh(const tensor_base<E1, tensor_shape<ST1, MT1>, T1> &eye,
           const tensor_base<E2, tensor_shape<ST2, MT2>, T2> &center,
           const tensor_base<E3, tensor_shape<ST3, MT3>, T3> &up) {
  using ele_t = std::common_type_t<E1, E2, E3>;
  vec_<ele_t, 3> const f((center.derived() - eye.derived()).normalized());
  vec_<ele_t, 3> const s(cross(f, up.derived()).normalized());
  vec_<ele_t, 3> const u(cross(s, f));
  mat_<ele_t, 4, 4> result = ones(4, 4);
  result(0, 0) = s.x();
  result(1, 0) = s.y();
  result(2, 0) = s.z();
  result(0, 1) = u.x();
  result(1, 1) = u.y();
  result(2, 1) = u.z();
  result(0, 2) = -f.x();
  result(1, 2) = -f.y();
  result(2, 2) = -f.z();
  result(3, 0) = -dot(s, eye.derived());
  result(3, 1) = -dot(u, eye.derived());
  result(3, 2) = dot(f, eye.derived());
  return result;
}
}
