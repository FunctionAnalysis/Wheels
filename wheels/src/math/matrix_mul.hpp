#pragma once

#include "../tensor/base.hpp"

namespace wheels {

// auto matrix_mul(ts1, ts2);
template <class EleT, class ShapeT, class A, class B, bool AIsMat, bool BIsMat>
class matrix_mul_result;
// matrix + matrix -> matrix
template <class EleT, class ShapeT, class A, class B>
class matrix_mul_result<EleT, ShapeT, A, B, true, true>
    : public tensor_op_result_base<
          EleT, ShapeT, binary_op_mul,
          matrix_mul_result<EleT, ShapeT, A, B, true, true>> {
public:
  using value_type = EleT;
  using shape_type = ShapeT;
  constexpr matrix_mul_result(A &&aa, B &&bb)
      : _a(forward<A>(aa)), _b(forward<B>(bb)) {}
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
// matrix + vector -> vector
template <class EleT, class ShapeT, class A, class B>
class matrix_mul_result<EleT, ShapeT, A, B, true, false>
    : public tensor_op_result_base<
          EleT, ShapeT, binary_op_mul,
          matrix_mul_result<EleT, ShapeT, A, B, true, false>> {
public:
  using value_type = EleT;
  using shape_type = ShapeT;
  constexpr matrix_mul_result(A &&aa, B &&bb)
      : _a(forward<A>(aa)), _b(forward<B>(bb)) {}
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
// vector + matrix -> vector
template <class EleT, class ShapeT, class A, class B>
class matrix_mul_result<EleT, ShapeT, A, B, false, true>
    : public tensor_op_result_base<
          EleT, ShapeT, binary_op_mul,
          matrix_mul_result<EleT, ShapeT, A, B, false, true>> {
public:
  using value_type = EleT;
  using shape_type = ShapeT;
  constexpr matrix_mul_result(A &&aa, B &&bb)
      : _a(forward<A>(aa)), _b(forward<B>(bb)) {}
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
struct overloaded<binary_op_mul,
                  category_tensor<E1, tensor_shape<ST1, MT1, NT1>, T1>,
                  category_tensor<E2, tensor_shape<ST2, MT2, NT2>, T2>> {
  template <class A, class B> constexpr auto operator()(A &&a, B &&b) const {
    assert(size_at(a, const_index<1>()) == size_at(b, const_index<0>()));
    using shape_t = std::decay_t<decltype(make_shape(
        size_at(a, const_index<0>()), size_at(b, const_index<1>())))>;
    return matrix_mul_result<std::common_type_t<E1, E2>, shape_t, A, B, true,
                             true>(forward<A>(a), forward<B>(b));
  }
};

template <class ST1, class MT1, class NT1, class E1, class T1, class ST2,
          class MT2, class E2, class T2>
struct overloaded<binary_op_mul,
                  category_tensor<E1, tensor_shape<ST1, MT1, NT1>, T1>,
                  category_tensor<E2, tensor_shape<ST2, MT2>, T2>> {
  template <class A, class B> constexpr auto operator()(A &&a, B &&b) const {
    assert(size_at(a, const_index<1>()) == size_at(b, const_index<0>()));
    using shape_t = tensor_shape<ST1, MT1>;
    return matrix_mul_result<std::common_type_t<E1, E2>, shape_t, A, B, true,
                             false>(forward<A>(a), forward<B>(b));
  }
};

template <class ST1, class MT1, class E1, class T1, class ST2, class MT2,
          class NT2, class E2, class T2>
struct overloaded<binary_op_mul,
                  category_tensor<E1, tensor_shape<ST1, MT1>, T1>,
                  category_tensor<E2, tensor_shape<ST2, MT2, NT2>, T2>> {
  template <class A, class B> constexpr auto operator()(A &&a, B &&b) const {
    assert(size_at(a, const_index<0>()) == size_at(b, const_index<0>()));
    using shape_t = tensor_shape<ST2, NT2>;
    return matrix_mul_result<std::common_type_t<E1, E2>, shape_t, A, B, false,
                             true>(forward<A>(a), forward<B>(b));
  }
};
}