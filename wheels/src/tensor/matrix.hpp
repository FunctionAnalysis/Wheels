#pragma once

#include "base.hpp"

#include "matrix_fwd.hpp"

namespace wheels {

// matrix base
template <class T> struct matrix_base : tensor_core<T> {
  constexpr auto rows() const { return this->size(const_index<0>()); }
  constexpr auto cols() const { return this->size(const_index<1>()); }

  constexpr decltype(auto) t() const & {
    return ::wheels::transpose(this->derived());
  }
  decltype(auto) t() & { return ::wheels::transpose(this->derived()); }
  decltype(auto) t() && {
    return ::wheels::transpose(std::move(this->derived()));
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
}
