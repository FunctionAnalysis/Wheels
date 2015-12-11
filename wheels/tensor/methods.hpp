#pragma once

#include "tensor.hpp"

namespace wheels {

// constant_result
template <class ShapeT, class ET>
class constant_result
    : public tensor_base<ShapeT, ET, constant_result<ShapeT, ET>> {
public:
  using shape_type = ShapeT;
  using value_type = ET;
  template <class EE>
  constexpr explicit constant_result(const ShapeT &s, EE &&v)
      : _shape(s), _val(forward<EE>(v)) {}
  const ShapeT &shape() const { return _shape; }
  const ET &value() const { return _val; }

private:
  ShapeT _shape;
  ET _val;
};

template <class ShapeT, class ET>
constexpr auto shape_of(const constant_result<ShapeT, ET> &t) {
  return t.shape();
}

template <class ShapeT, class ET, class... SubTs>
constexpr const ET &element_at(const constant_result<ShapeT, ET> &t,
                               const SubTs &... subs) {
  return t.value();
}

template <class ShapeT, class ET, class IndexT>
constexpr const ET &element_at_index(const constant_result<ShapeT, ET> &t,
                                     const IndexT &ind) {
  return t.value();
}

template <class FunT, class ShapeT, class ET, class... Ts>
void for_each_nonzero_element(FunT &&fun, const constant_result<ShapeT, ET> &t,
                              Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  if (t.value()) {
    for_each_element(forward<FunT>(fun), t, forward<Ts>(ts)...);
  }
}

template <class ShapeT, class ET>
constexpr bool all_of(const constant_result<ShapeT, ET> &t) {
  return !!t.value();
}

template <class ShapeT, class ET>
constexpr bool any_of(const constant_result<ShapeT, ET> &t) {
  return !!t.value();
}

// constants
template <class ET, class ST, class... SizeTs>
constexpr auto constants(const tensor_shape<ST, SizeTs...> &shape, ET &&v) {
  return constant_result<tensor_shape<ST, SizeTs...>, std::decay_t<ET>>(
      shape, forward<ET>(v));
}

// zeros
template <class ET = double, class ST, class... SizeTs>
constexpr auto zeros(const tensor_shape<ST, SizeTs...> &shape) {
  return constants<ET>(shape, 0);
}
template <class ET = double, class... SizeTs>
constexpr auto zeros(const SizeTs &... sizes) {
  return constants<ET>(make_shape(sizes...), 0);
}

// ones
template <class ET = double, class ST, class... SizeTs>
constexpr auto ones(const tensor_shape<ST, SizeTs...> &shape) {
  return constants<ET>(shape, 1);
}
template <class ET = double, class... SizeTs>
constexpr auto ones(const SizeTs &... sizes) {
  return constants<ET>(make_shape(sizes...), 1);
}

// fast implementation of ewise ops
// all constants
template <class OpT, class ShapeT, class EleT, class... ShapeTs, class... EleTs>
struct overloaded<
    OpT, category_tensor<ShapeT, EleT, constant_result<ShapeT, EleT>>,
    category_tensor<ShapeTs, EleTs, constant_result<ShapeTs, EleTs>>...> {
  template <class TT, class... TTs>
  constexpr decltype(auto) operator()(TT &&t, TTs &&... ts) const {
    assert(all_same(shape_of(t), shape_of(ts)...));
    return constants(t.shape(), OpT()(t.value(), ts.value()...));
  }
};
template <class ShapeT, class EleT, class ShapeT2, class EleT2>
struct overloaded<
    binary_op_mul, category_tensor<ShapeT, EleT, constant_result<ShapeT, EleT>>,
    category_tensor<ShapeT2, EleT2, constant_result<ShapeT2, EleT2>>> {
  template <class TT, class TT2>
  constexpr int operator()(TT &&t, TT2 &&t2) const {
    static_assert(always<bool, false, TT, TT2>::value,
                  "use ewise_mul(t1, t2) if you want to compute element-wise "
                  "product of two tensors");
  }
};
template <class ShapeT, class EleT, class ShapeT2, class EleT2>
struct overloaded<
    ewised<binary_op_mul>,
    category_tensor<ShapeT, EleT, constant_result<ShapeT, EleT>>,
    category_tensor<ShapeT2, EleT2, constant_result<ShapeT2, EleT2>>> {
  template <class TT, class TT2>
  constexpr decltype(auto) operator()(TT &&t, TT2 &&t2) const {
    assert(all_same(shape_of(t), shape_of(t2)));
    return constants(t.shape(), t.value() * t2.value());
  }
};

// tensor vs scalar
template <class OpT, class ShapeT, class EleT>
struct overloaded<
    OpT, category_tensor<ShapeT, EleT, constant_result<ShapeT, EleT>>, void> {
  template <class T1, class T2>
  constexpr decltype(auto) operator()(T1 &&t1, T2 &&t2) const {
    return constants(t1.shape(), OpT()(t1.value(), forward<T2>(t2)));
  }
};
// scalar vs tensor
template <class OpT, class ShapeT, class EleT>
struct overloaded<
    OpT, void, category_tensor<ShapeT, EleT, constant_result<ShapeT, EleT>>> {
  template <class T1, class T2>
  constexpr decltype(auto) operator()(T1 &&t1, T2 &&t2) const {
    return constants(t2.shape(), OpT()(forward<T1>(t1), t2.value()));
  }
};

// eye

// iota

// meshgrid

// ewise ops
template <class ShapeT, class EleT, class OpT, class InputT, class... InputTs>
class ewise_op_result
    : public tensor_base<ShapeT, EleT, ewise_op_result<ShapeT, EleT, OpT,
                                                       InputT, InputTs...>> {
public:
  using shape_type = ShapeT;
  using value_type = EleT;
  constexpr explicit ewise_op_result(OpT o, InputT &&in, InputTs &&... ins)
      : op(o), inputs(forward<InputT>(in), forward<InputTs>(ins)...) {}

public:
  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(op, inputs);
  }
  template <class V> decltype(auto) fields(V &&visitor) const {
    return visitor(op, inputs);
  }

public:
  OpT op;
  std::tuple<InputT, InputTs...> inputs;
};
template <class ShapeT, class ET, class OpT, class InputT, class... InputTs>
constexpr ewise_op_result<ShapeT, ET, OpT, InputT, InputTs...>
make_ewise_op_result(OpT op, InputT &&input, InputTs &&... inputs) {
  return ewise_op_result<ShapeT, ET, OpT, InputT, InputTs...>(
      op, forward<InputT>(input), forward<InputTs>(inputs)...);
}

// shape_of
template <class ShapeT, class EleT, class OpT, class InputT, class... InputTs>
constexpr ShapeT
shape_of(const ewise_op_result<ShapeT, EleT, OpT, InputT, InputTs...> &ts) {
  return shape_of(std::get<0>(ts.inputs));
}

// element_at
namespace details {
template <class EwiseOpResultT, size_t... Is, class... SubTs>
constexpr decltype(auto)
_element_at_ewise_op_result_seq(EwiseOpResultT &ts,
                                const const_ints<size_t, Is...> &,
                                const SubTs &... subs) {
  return ts.op(element_at(std::get<Is>(ts.inputs), subs...)...);
}
}
template <class ShapeT, class EleT, class OpT, class InputT, class... InputTs,
          class... SubTs>
constexpr decltype(auto)
element_at(const ewise_op_result<ShapeT, EleT, OpT, InputT, InputTs...> &ts,
           const SubTs &... subs) {
  return details::_element_at_ewise_op_result_seq(
      ts, make_const_sequence_for<InputT, InputTs...>(), subs...);
}

// shortcuts
namespace details {
template <class EwiseOpResultT, size_t... Is, class IndexT>
constexpr decltype(auto)
_element_at_index_ewise_op_result_seq(EwiseOpResultT &ts,
                                      const const_ints<size_t, Is...> &,
                                      const IndexT &index) {
  return ts.op(element_at_index(std::get<Is>(ts.inputs), index)...);
}
}
template <class ShapeT, class EleT, class OpT, class InputT, class... InputTs,
          class IndexT>
constexpr decltype(auto) element_at_index(
    const ewise_op_result<ShapeT, EleT, OpT, InputT, InputTs...> &ts,
    const IndexT &index) {
  return details::_element_at_index_ewise_op_result_seq(
      ts, make_const_sequence_for<InputT, InputTs...>(), index);
}

// other ops are overloaded as ewise operation results defaultly
// all tensors
template <class OpT, class ShapeT, class EleT, class T, class... ShapeTs,
          class... EleTs, class... Ts>
struct overloaded<OpT, category_tensor<ShapeT, EleT, T>,
                  category_tensor<ShapeTs, EleTs, Ts>...> {
  template <class TT, class... TTs>
  constexpr decltype(auto) operator()(TT &&t, TTs &&... ts) const {
    assert(all_same(shape_of(t), shape_of(ts)...));
    using ele_t = std::decay_t<decltype(
        OpT()(std::declval<EleT>(), std::declval<EleTs>()...))>;
    return make_ewise_op_result<ShapeT, ele_t>(OpT(), forward<TT>(t),
                                               forward<TTs>(ts)...);
  }
};

template <class ShapeT, class EleT, class T, class ShapeT2, class EleT2,
          class T2>
struct overloaded<binary_op_mul, category_tensor<ShapeT, EleT, T>,
                  category_tensor<ShapeT2, EleT2, T2>> {
  template <class TT, class TT2>
  constexpr int operator()(TT &&t, TT2 &&t2) const {
    static_assert(always<bool, false, TT, TT2>::value,
                  "use ewise_mul(t1, t2) if you want to compute element-wise "
                  "product of two tensors");
  }
};

template <class ShapeT, class EleT, class T, class ShapeT2, class EleT2,
          class T2>
struct overloaded<ewised<binary_op_mul>, category_tensor<ShapeT, EleT, T>,
                  category_tensor<ShapeT2, EleT2, T2>> {
  template <class TT, class TT2>
  constexpr decltype(auto) operator()(TT &&t, TT2 &&t2) const {
    using ele_t = decltype(std::declval<EleT>() * std::declval<EleT2>());
    return make_ewise_op_result<ShapeT, ele_t>(binary_op_mul(), forward<TT>(t),
                                               forward<TT2>(t2));
  }
};

// tensor vs scalar
template <class OpT, class ShapeT, class EleT, class T>
struct overloaded<OpT, category_tensor<ShapeT, EleT, T>, void> {
  template <class T1, class T2>
  constexpr decltype(auto) operator()(T1 &&t1, T2 &&t2) const {
    using ele_t =
        std::decay_t<decltype(OpT()(std::declval<EleT>(), std::declval<T2>()))>;
    return make_ewise_op_result<ShapeT, ele_t>(
        OpT()(const_symbol<0>(), forward<T2>(t2)), forward<T1>(t1));
  }
};
// scalar vs tensor
template <class OpT, class ShapeT, class EleT, class T>
struct overloaded<OpT, void, category_tensor<ShapeT, EleT, T>> {
  template <class T1, class T2>
  constexpr decltype(auto) operator()(T1 &&t1, T2 &&t2) const {
    using ele_t =
        std::decay_t<decltype(OpT()(std::declval<T1>(), std::declval<EleT>()))>;
    return make_ewise_op_result<ShapeT, ele_t>(
        OpT()(forward<T1>(t1), const_symbol<0>()), forward<T2>(t2));
  }
};
// tensor vs const_expr
template <class OpT, class ShapeT, class EleT, class T>
struct overloaded<OpT, category_tensor<ShapeT, EleT, T>, category_const_expr> {
  template <class T1, class T2>
  constexpr decltype(auto) operator()(T1 &&t1, T2 &&t2) const {
    return const_binary_op<OpT, const_coeff<std::decay_t<T1>>, T2>(
        OpT(), as_const_coeff(forward<T1>(t1)), forward<T2>(t2));
  }
};
// const_expr vs tensor
template <class OpT, class ShapeT, class EleT, class T>
struct overloaded<OpT, category_const_expr, category_tensor<ShapeT, EleT, T>> {
  template <class T1, class T2>
  constexpr decltype(auto) operator()(T1 &&t1, T2 &&t2) const {
    return const_binary_op<OpT, T1, const_coeff<std::decay_t<T2>>>(
        OpT(), forward<T1>(t1), as_const_coeff(forward<T2>(t2)));
  }
};

// auto matrix_mul(ts1, ts2);
template <class ShapeT, class EleT, class A, class B, bool AIsMat, bool BIsMat>
class matrix_mul_result;
// matrix + matrix -> matrix
template <class ShapeT, class EleT, class A, class B>
class matrix_mul_result<ShapeT, EleT, A, B, true, true>
    : public tensor_base<ShapeT, EleT,
                         matrix_mul_result<ShapeT, EleT, A, B, true, true>> {
public:
  using value_type = EleT;
  using shape_type = ShapeT;
  constexpr matrix_mul_result(A &&aa, B &&bb)
      : _a(forward<A>(aa)), _b(forward<B>(bb)) {}
  constexpr auto shape() const {
    return make_shape(size_at(_a, const_index<0>()),
                      size_at(b, const_index<1>()));
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
template <class ShapeT, class EleT, class A, class B>
class matrix_mul_result<ShapeT, EleT, A, B, true, false>
    : public tensor_base<ShapeT, EleT,
                         matrix_mul_result<ShapeT, EleT, A, B, true, false>> {
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
template <class ShapeT, class EleT, class A, class B>
class matrix_mul_result<ShapeT, EleT, A, B, false, true>
    : public tensor_base<ShapeT, EleT,
                         matrix_mul_result<ShapeT, EleT, A, B, false, true>> {
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
template <class ShapeT, class EleT, class A, class B, bool AIsMat, bool BIsMat>
constexpr auto
shape_of(const matrix_mul_result<ShapeT, EleT, A, B, AIsMat, BIsMat> &m) {
  return m.shape();
}
// element_at
template <class ShapeT, class EleT, class A, class B, bool AIsMat, bool BIsMat,
          class... SubTs>
constexpr decltype(auto)
element_at(const matrix_mul_result<ShapeT, EleT, A, B, AIsMat, BIsMat> &m,
           const SubTs &... subs) {
  return m.at_subs(subs...);
}

template <class ST1, class MT1, class NT1, class E1, class T1, class ST2,
          class MT2, class NT2, class E2, class T2>
struct overloaded<binary_op_mul,
                  category_tensor<tensor_shape<ST1, MT1, NT1>, E1, T1>,
                  category_tensor<tensor_shape<ST2, MT2, NT2>, E2, T2>> {
  template <class A, class B> constexpr auto operator()(A &&a, B &&b) const {
    assert(size_at(a, const_index<1>()) == size_at(b, const_index<0>()));
    using shape_t = std::decay_t<decltype(make_shape(
        size_at(a, const_index<0>()), size_at(b, const_index<1>())))>;
    return matrix_mul_result<shape_t, std::common_type_t<E1, E2>, A, B, true,
                             true>(forward<A>(a), forward<B>(b));
  }
};

template <class ST1, class MT1, class NT1, class E1, class T1, class ST2,
          class MT2, class E2, class T2>
struct overloaded<binary_op_mul,
                  category_tensor<tensor_shape<ST1, MT1, NT1>, E1, T1>,
                  category_tensor<tensor_shape<ST2, MT2>, E2, T2>> {
  template <class A, class B> constexpr auto operator()(A &&a, B &&b) const {
    assert(size_at(a, const_index<1>()) == size_at(b, const_index<0>()));
    using shape_t = tensor_shape<ST1, MT1>;
    return matrix_mul_result<shape_t, std::common_type_t<E1, E2>, A, B, true,
                             false>(forward<A>(a), forward<B>(b));
  }
};

template <class ST1, class MT1, class E1, class T1, class ST2, class MT2,
          class NT2, class E2, class T2>
struct overloaded<binary_op_mul,
                  category_tensor<tensor_shape<ST1, MT1>, E1, T1>,
                  category_tensor<tensor_shape<ST2, MT2, NT2>, E2, T2>> {
  template <class A, class B> constexpr auto operator()(A &&a, B &&b) const {
    assert(size_at(a, const_index<0>()) == size_at(b, const_index<0>()));
    using shape_t = tensor_shape<ST2, NT2>;
    return matrix_mul_result<shape_t, std::common_type_t<E1, E2>, A, B, false,
                             true>(forward<A>(a), forward<B>(b));
  }
};

// transpose
template <class ShapeT, class ET, class T>
class matrix_transpose
    : public tensor_base<ShapeT, ET, matrix_transpose<ShapeT, ET, T>> {
public:
  using value_type = ET;
  using shape_type = ShapeT;
  constexpr explicit matrix_transpose(T &&in) : _input(forward<T>(in)) {}
  constexpr auto shape() const {
    return make_shape(size_at(_input, const_index<1>()),
                      size_at(_input, const_index<0>()));
  }
  template <class SubT1, class SubT2>
  constexpr decltype(auto) at_subs(const SubT1 &s1, const SubT2 &s2) const {
    return element_at(_input, s2, s1);
  }

private:
  T _input;
};

// shape_of
template <class ShapeT, class ET, class T>
constexpr auto shape_of(const matrix_transpose<ShapeT, ET, T> &m) {
  return m.shape();
}
// element_at
template <class ShapeT, class ET, class T, class SubT1, class SubT2>
constexpr decltype(auto) element_at(const matrix_transpose<ShapeT, ET, T> &m,
                                    const SubT1 &s1, const SubT2 &s2) {
  return m.at_subs(s1, s2);
}

template <class ST, class MT, class NT, class ET, class T>
constexpr auto
transpose(const tensor_base<tensor_shape<ST, MT, NT>, ET, T> &t) {
  return matrix_transpose<tensor_shape<ST, NT, MT>, ET, const T &>(t.derived());
}
template <class ST, class MT, class NT, class ET, class T>
constexpr auto transpose(tensor_base<tensor_shape<ST, MT, NT>, ET, T> &t) {
  return matrix_transpose<tensor_shape<ST, NT, MT>, ET, T &>(t.derived());
}
template <class ST, class MT, class NT, class ET, class T>
constexpr auto transpose(tensor_base<tensor_shape<ST, MT, NT>, ET, T> &&t) {
  return matrix_transpose<tensor_shape<ST, NT, MT>, ET, T>(
      std::move(t.derived()));
}
template <class ST, class MT, class NT, class ET, class T>
constexpr auto
transpose(const tensor_base<tensor_shape<ST, MT, NT>, ET, T> &&t) {
  return matrix_transpose<tensor_shape<ST, NT, MT>, ET, const T>(
      std::move(t.derived()));
}

// distance
template <class ShapeT1, class ET1, class T1, class ShapeT2, class ET2,
          class T2>
constexpr auto distance(const tensor_base<ShapeT1, ET1, T1> &t1,
                        const tensor_base<ShapeT2, ET2, T2> &t2) {
  return norm(t1.derived() - t2.derived());
}

// dot(ts1, ts2);
template <class ShapeT1, class ET1, class T1, class ShapeT2, class ET2,
          class T2>
auto dot(const tensor_base<ShapeT1, ET1, T1> &t1,
         const tensor_base<ShapeT2, ET2, T2> &t2) {
  using result_t = std::common_type_t<ET1, ET2>;
  assert(shape_of(t1.derived()) == shape_of(t2.derived()));
  result_t result = 0.0;
  for_each_element([&result](auto &&e1, auto &&e2) { result += e1 * e2; },
                   t1.derived(), t2.derived());
  return result;
}

// auto cross(ts1, ts2);
template <class ST1, class NT1, class E1, class T1, class ST2, class NT2,
          class E2, class T2>
constexpr auto cross(const tensor_base<tensor_shape<ST1, NT1>, E1, T1> &a,
                     const tensor_base<tensor_shape<ST2, NT2>, E2, T2> &b) {
  using result_t = std::common_type_t<E1, E2>;
  return vec_<result_t, 3>(a.y() * b.z() - a.z() * b.y(),
                           a.z() * b.x() - a.x() * b.z(),
                           a.x() * b.y() - a.y() * b.x());
}
}