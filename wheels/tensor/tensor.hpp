#pragma once

#include <cassert>

#include "../core/const_expr.hpp"
#include "../core/constants.hpp"
#include "../core/overloads.hpp"
#include "../core/serialize.hpp"
#include "../core/types.hpp"

#include "shape.hpp"

namespace wheels {

// base of all tensor types
// tensor_core
template <class T> struct tensor_core {
  const tensor_core &core() const { return *this; }
  constexpr const T &derived() const { return static_cast<const T &>(*this); }
  T &derived() { return static_cast<T &>(*this); }

  constexpr auto shape() const { return wheels::shape_of(derived()); }
  template <class K, K Idx>
  constexpr auto size(const const_ints<K, Idx> &i) const {
    return wheels::size_at(derived(), i);
  }
  constexpr auto numel() const { return wheels::numel(derived()); }

  constexpr auto norm_squared() const {
    return wheels::norm_squared(derived());
  }
  constexpr auto norm() const { return wheels::norm(derived()); }
  constexpr auto normalized() const & { return derived() / this->norm(); }
  auto normalized() && { return std::move(derived()) / this->norm(); }

  template <class... SubTs>
  constexpr decltype(auto) operator()(const SubTs &... subs) const {
    return element_at(derived(), subs...);
  }
  template <class IndexT>
  constexpr decltype(auto) operator[](const IndexT &ind) const {
    return element_at_index(derived(), ind);
  }
  template <class... SubTs> decltype(auto) operator()(const SubTs &... subs) {
    return element_at(derived(), subs...);
  }
  template <class IndexT> decltype(auto) operator[](const IndexT &ind) {
    return element_at_index(derived(), ind);
  }
};

template <class ShapeT, class ET> class tensor;

// tensor_base
template <class ShapeT, class ET, class T> struct tensor_base : tensor_core<T> {
  using shape_type = ShapeT;
  static constexpr size_t rank = ShapeT::rank;
  using value_type = ET;

  const tensor_base &base() const { return *this; }

  static constexpr auto type_of_shape() { return types<ShapeT>(); }
  static constexpr auto type_of_element() { return types<ET>(); }

  constexpr tensor<ShapeT, ET> eval() const {
    return tensor<ShapeT, ET>(derived());
  }
  constexpr operator tensor<ShapeT, ET>() const { return eval(); }
};

// for vectors
template <class ST, class NT, class ET, class T>
struct tensor_base<tensor_shape<ST, NT>, ET, T> : tensor_core<T> {
  using shape_type = tensor_shape<ST, NT>;
  static constexpr size_t rank = 1;
  using value_type = ET;

  const tensor_base &base() const { return *this; }

  static constexpr auto type_of_shape() { return types<ShapeT>(); }
  static constexpr auto type_of_element() { return types<ET>(); }

  constexpr tensor<tensor_shape<ST, NT>, ET> eval() const {
    return tensor<tensor_shape<ST, NT>, ET>(derived());
  }
  constexpr operator tensor<tensor_shape<ST, NT>, ET>() const { return eval(); }

  template <class ST2, class NT2, class ET2, class T2>
  constexpr decltype(auto)
  dot(const tensor_base<tensor_shape<ST2, NT2>, ET2, T2> &t) const {
    return wheels::dot(*this, t);
  }
  template <class ST2, class NT2, class ET2, class T2>
  constexpr decltype(auto)
  cross(const tensor_base<tensor_shape<ST2, NT2>, ET2, T2> &t) const {
    return wheels::cross(*this, t);
  }
};

// for matrices
template <class ST, class MT, class NT, class ET, class T>
struct tensor_base<tensor_shape<ST, MT, NT>, ET, T> : tensor_core<T> {
  using shape_type = tensor_shape<ST, MT, NT>;
  static constexpr size_t rank = 2;
  using value_type = ET;

  const tensor_base &base() const { return *this; }

  static constexpr auto type_of_shape() {
    return types<tensor_shape<ST, MT, NT>>();
  }
  static constexpr auto type_of_element() { return types<ET>(); }

  constexpr tensor<tensor_shape<ST, MT, NT>, ET> eval() const {
    return tensor<tensor_shape<ST, MT, NT>, ET>(derived());
  }
  constexpr operator tensor<tensor_shape<ST, MT, NT>, ET>() const {
    return eval();
  }

  constexpr auto t() const & { return transpose(derived()); }
  auto t() & { return transpose(derived()); }
  auto t() && { return transpose(std::move(derived())); }
};

// category_for_overloading
// except fields(...)
template <class ShapeT, class ET, class T, class OpT>
constexpr auto category_for_overloading(const tensor_base<ShapeT, ET, T> &,
                                        const common_func<OpT> &) {
  return category_tensor<ShapeT, ET, T>();
}

// -- necessary tensor functions
// Shape shape_of(ts);
template <class T>
constexpr tensor_shape<size_t> shape_of(const tensor_core<T> &) {
  static_assert(always<bool, false, T>::value,
                "shape_of(const T &) is not supported by tensor_core<T>, do "
                "you forget to call .derived()?");
}

// Scalar element_at(ts, subs ...);
template <class T, class... SubTs>
constexpr double element_at(const tensor_core<T> &t, const SubTs &...) {
  static_assert(always<bool, false, T>::value,
                "element_at(const T &) is not supported by tensor_core<T>, do "
                "you forget to call .derived()?");
}
template <class T, class... SubTs>
constexpr double &element_at(tensor_core<T> &t, const SubTs &...) {
  static_assert(always<bool, false, T>::value,
                "element_at(T &) is not supported by tensor_core<T>, do you "
                "forget to call .derived()?");
}

// -- auxiliary tensor functions
// auto size_at(ts, const_int);
template <class T, class ST, ST Idx>
constexpr auto size_at(const tensor_core<T> &t,
                       const const_ints<ST, Idx> &idx) {
  return shape_of(t.derived()).at(idx);
}

// auto numel(ts)
template <class T> constexpr auto numel(const tensor_core<T> &t) {
  return shape_of(t.derived()).magnitude();
}

// Scalar element_at_index(ts, index);
template <class T, class IndexT>
constexpr decltype(auto) element_at_index(const tensor_core<T> &t,
                                          const IndexT &ind) {
  return invoke_with_subs(shape_of(t.derived()), ind, [&t](auto &&... subs) {
    return element_at(t.derived(), subs...);
  });
}

// void reserve_shape(ts, shape);
template <class T, class ST, class... SizeTs>
void reserve_shape(tensor_core<T> &, const tensor_shape<ST, SizeTs...> &shape) {
}

// for_each_element
template <class FunT, class T, class... Ts>
void for_each_element(FunT &&fun, T &&t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for_each_subscript(shape_of(t), [&](auto &&... subs) {
    fun(element_at(t, subs...), element_at(ts, subs...)...);
  });
}

// for_each_element_if
template <class FunT, class T, class... Ts>
bool for_each_element_if(FunT &&fun, T &&t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for_each_subscript_if(shape_of(t), [&](auto &&... subs) {
    return fun(element_at(t, subs...), element_at(ts, subs...)...);
  });
}

// for_each_nonzero_element
template <class FunT, class T, class... Ts>
void for_each_nonzero_element(FunT &&fun, T &&t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for_each_subscript(shape_of(t), [&](auto &&... subs) {
    decltype(auto) e = element_at(t, subs...);
    if (e) {
      fun(e, element_at(ts, subs...)...);
    }
  });
}

// void assign_elements(to, from);
template <class To, class From> void assign_elements(To &to, const From &from) {
  decltype(auto) s = shape_of(from);
  if (shape_of(to) != s) {
    reserve_shape(to, s);
  }
  for_each_element([](auto &to_e, const auto from_e) { to_e = from_e; }, to,
                   from);
}

// Scalar reduce_elements(ts, initial, functor);
template <class T, class E, class ReduceT>
E reduce_elements(const T &t, E initial, ReduceT &&red) {
  for_each_element([&initial, &red](auto &&e) { initial = red(initial, e); },
                   t);
  return initial;
}

// Scalar norm_squared(ts)
template <class ShapeT, class ET, class T>
ET norm_squared(const tensor_base<ShapeT, ET, T> &t) {
  ET result = 0.0;
  for_each_nonzero_element([&result](auto &&e) { result += e * e; }, t.derived());
  return result;
}

// Scalar norm(ts)
template <class ShapeT, class ET, class T>
constexpr ET norm(const tensor_base<ShapeT, ET, T> &t) {
  return sqrt(norm_squared(t.derived()));
}

// bool all(s)
template <class ShapeT, class ET, class T>
constexpr bool all_of(const tensor_base<ShapeT, ET, T> &t) {
  return for_each_element_if([](auto &&e) { return !!e; });
}

// bool any(s)
template <class ShapeT, class ET, class T>
constexpr bool any_of(const tensor_base<ShapeT, ET, T> &t) {
  return !for_each_element_if([](auto &&e) { return !e; });
}

// -- special tensor functions
//// ewise_ops
template <class ShapeT, class EleT, class OpT, class InputT, class... InputTs>
class ewise_op_result
    : public tensor_base<ShapeT, EleT, ewise_op_result<ShapeT, EleT, OpT,
                                                       InputT, InputTs...>> {
public:
  using shape_type = ShapeT;
  using value_type = EleT;
  constexpr explicit ewise_op_result(OpT o, InputT &&in, InputTs &&... ins)
      : op(o), inputs(forward<InputT>(in), forward<InputTs>(ins)...) {}
  template <wheels_enable_if((std::is_same<EleT, bool>::value))>
  constexpr operator bool() const {
    return reduce_elements(*this, true, binary_op_and());
  }

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

template <class T1, class T2>
constexpr decltype(auto) ewise_mul(T1 &&t1, T2 &&t2) {
  assert(shape_of(t1) == shape_of(t2));
  using shape_t = type_t(t1.type_of_shape());
  using ele_t1 = type_t(t1.type_of_element());
  using ele_t2 = type_t(t2.type_of_element());
  using ele_t = decltype(std::declval<ele_t1>() * std::declval<ele_t2>());
  return make_ewise_op_result<shape_t, ele_t>(binary_op_mul(), forward<T1>(t1),
                                              forward<T2>(t2));
}

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

// auto normalize(ts)
template <class T> constexpr auto normalize(T &&t) {
  return forward<T>(t) / norm(t);
}

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

// tensor_storage
constexpr struct _with_elements {
} with_elements;
constexpr struct _with_iterators {
} with_iterators;

template <class ShapeT, class ET, class T, bool StaticShape>
class tensor_storage;
namespace details {
template <class T, size_t N, size_t... Is>
constexpr auto _init_std_array_seq(const_ints<size_t, Is...>) {
  return std::array<T, N>{{(T)always<int, 0, const_index<Is>>::value...}};
}
template <class T, size_t N> constexpr auto _init_std_array() {
  return _init_std_array_seq<T, N>(make_const_sequence(const_size<N>()));
}
}
template <class ShapeT, class ET, class T>
class tensor_storage<ShapeT, ET, T, true> : public tensor_base<ShapeT, ET, T> {
public:
  using shape_type = ShapeT;
  using value_type = ET;

public:
  constexpr tensor_storage()
      : _data(details::_init_std_array<value_type,
                                       shape_type::static_magnitude>()) {}
  template <class... EleTs>
  constexpr tensor_storage(const shape_type &shape, const _with_elements &,
                           EleTs &&... eles)
      : _data{{(value_type)forward<EleTs>(eles)...}} {}
  template <class IterT>
  tensor_storage(const shape_type &shape, const _with_iterators &, IterT begin,
                 IterT end) {
    std::copy(begin, end, _data.begin());
  }

  constexpr tensor_storage(const tensor_storage &) = default;
  tensor_storage(tensor_storage &&) = default;
  tensor_storage &operator=(const tensor_storage &) = default;
  tensor_storage &operator=(tensor_storage &&) = default;

  constexpr auto shape() const { return shape_type(); }
  constexpr const auto &data() const { return _data; }
  auto &data() { return _data; }

public:
  template <class ArcT> void serialize(ArcT &ar) { ar(_data); }
  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(_data);
  }
  template <class V> constexpr decltype(auto) fields(V &&visitor) const {
    return visitor(_data);
  }

private:
  std::array<value_type, shape_type::static_magnitude> _data;
};

template <class ShapeT, class ET, class T>
class tensor_storage<ShapeT, ET, T, false> : public tensor_base<ShapeT, ET, T> {
public:
  using shape_type = ShapeT;
  using value_type = ET;

public:
  constexpr tensor_storage() {}
  template <class... EleTs>
  constexpr tensor_storage(const shape_type &shape, const _with_elements &,
                           EleTs &&... eles)
      : _shape(shape), _data({(value_type)forward<EleTs>(eles)...}) {}
  template <class IterT>
  tensor_storage(const shape_type &shape, const _with_iterators &, IterT begin,
                 IterT end)
      : _shape(shape), _data(begin, end) {}

  constexpr tensor_storage(const tensor_storage &) = default;
  tensor_storage(tensor_storage &&) = default;
  tensor_storage &operator=(const tensor_storage &) = default;
  tensor_storage &operator=(tensor_storage &&) = default;

  constexpr const auto &shape() const { return _shape; }
  auto &shape() { return _shape; }
  const auto &data() const { return _data; }
  auto &data() { return _data; }

public:
  template <class ArcT> void serialize(ArcT &ar) { ar(_shape, _data); }
  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(_shape, _data);
  }
  template <class V> constexpr decltype(auto) fields(V &&visitor) const {
    return visitor(_shape, _data);
  }

private:
  shape_type _shape;
  std::vector<value_type> _data;
};

namespace details {
template <class ShapeT, size_t... Is>
constexpr ShapeT _make_shape_from_magnitude_seq(size_t magnitude,
                                                const_ints<size_t, Is...>) {
  static_assert(ShapeT::dynamic_size_num == 1,
                "ShapeT::dynamic_size_num should be 1 here");
  static_assert(ShapeT::last_dynamic_dim >= 0,
                "ShapeT::last_dynamic_dim is not valid");
  return ShapeT(conditional(const_bool<Is == ShapeT::last_dynamic_dim>(),
                            magnitude / ShapeT::static_magnitude,
                            std::ignore)...);
}
}

template <class ShapeT, class ET>
class tensor
    : public tensor_storage<ShapeT, ET, tensor<ShapeT, ET>, ShapeT::is_static> {
  using base_t =
      tensor_storage<ShapeT, ET, tensor<ShapeT, ET>, ShapeT::is_static>;

public:
  using value_type = ET;
  using shape_type = ShapeT;

  constexpr tensor() : base_t() {}

  template <class... EleTs, class = std::enable_if_t<(
                                ShapeT::dynamic_size_num == 0 &&
                                all(std::is_convertible<EleTs, ET>::value...))>>
  constexpr tensor(EleTs &&... eles)
      : base_t(ShapeT(), with_elements, forward<EleTs>(eles)...) {}

  template <
      class... EleTs, class = void,
      class = std::enable_if_t<(ShapeT::dynamic_size_num == 1 &&
                                all(std::is_convertible<EleTs, ET>::value...))>>
  constexpr tensor(EleTs &&... eles)
      : base_t(details::_make_shape_from_magnitude_seq<ShapeT>(
                   sizeof...(EleTs),
                   make_const_sequence(const_size<ShapeT::rank>())),
               with_elements, forward<EleTs>(eles)...) {}

  template <class... EleTs>
  constexpr tensor(const ShapeT &shape, const _with_elements &we,
                   EleTs &&... eles)
      : base_t(shape, we, forward<EleTs>(eles)...) {}

  template <class = std::enable_if_t<(ShapeT::dynamic_size_num == 0)>>
  constexpr tensor(std::initializer_list<value_type> ilist)
      : base_t(ShapeT(), with_iterators, ilist.begin(), ilist.end()) {}

  template <class = void,
            class = std::enable_if_t<(ShapeT::dynamic_size_num == 1)>>
  constexpr tensor(std::initializer_list<value_type> ilist)
      : base_t(
            details::_make_shape_from_magnitude_seq<ShapeT>(
                ilist.size(), make_const_sequence(const_size<ShapeT::rank>())),
            with_iterators, ilist.begin(), ilist.end()) {}

  constexpr tensor(const ShapeT &shape, std::initializer_list<value_type> ilist)
      : base_t(shape, with_iterators, ilist.begin(), ilist.end()) {}

  template <class IterT,
            class = std::enable_if_t<(ShapeT::dynamic_size_num == 0 &&
                                      is_iterator<IterT>::value)>>
  constexpr tensor(IterT begin, IterT end, yes * = nullptr)
      : base_t(ShapeT(), with_iterators, begin, end) {}

  template <class IterT,
            class = std::enable_if_t<(ShapeT::dynamic_size_num == 1 &&
                                      is_iterator<IterT>::value)>>
  constexpr tensor(IterT begin, IterT end, no * = nullptr)
      : base_t(details::_make_shape_from_magnitude_seq<ShapeT>(
                   std::distance(begin, end),
                   make_const_sequence(const_size<ShapeT::rank>())),
               with_iterators, begin, end) {}

  template <class IterT, class = std::enable_if_t<is_iterator<IterT>::value>>
  constexpr tensor(const ShapeT &shape, IterT begin, IterT end)
      : base_t(shape, with_iterators, begin, end) {}

  tensor(const tensor &) = default;
  tensor(tensor &&) = default;
  tensor &operator=(const tensor &) = default;
  tensor &operator=(tensor &&) = default;

  template <class AnotherT>
  constexpr tensor(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
  }
  template <class AnotherT>
  tensor &operator=(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
    return *this;
  }

public:
  constexpr decltype(auto) shape() const { return base_t::shape(); }
  template <class... SubTs>
  constexpr decltype(auto) operator()(const SubTs &... subs) const {
    static_assert(sizeof...(SubTs) == ShapeT::rank,
                  "invalid number of subscripts");
    return base_t::data()[sub2ind(shape(), subs...)];
  }
  template <class... SubTs> decltype(auto) operator()(const SubTs &... subs) {
    static_assert(sizeof...(SubTs) == ShapeT::rank,
                  "invalid number of subscripts");
    return base_t::data()[sub2ind(shape(), subs...)];
  }
  template <class IndexT>
  constexpr decltype(auto) operator[](const IndexT &ind) const {
    return base_t::data()[ind];
  }
  template <class IndexT> decltype(auto) operator[](const IndexT &ind) {
    return base_t::data()[ind];
  }
};

// necessary
template <class ET, class ShapeT>
constexpr auto shape_of(const tensor<ShapeT, ET> &t) {
  return t.shape();
}
template <class ET, class ShapeT, class... SubTs>
constexpr decltype(auto) element_at(const tensor<ShapeT, ET> &t,
                                    const SubTs &... subs) {
  return t(subs...);
}
template <class ET, class ShapeT, class... SubTs>
decltype(auto) element_at(tensor<ShapeT, ET> &t, const SubTs &... subs) {
  return t(subs...);
}

// auxiliary
template <class ET, class ShapeT, class IndexT>
constexpr decltype(auto) element_at_index(const tensor<ShapeT, ET> &t,
                                          const IndexT &ind) {
  return t[ind];
}
template <class ET, class ShapeT, class IndexT>
decltype(auto) element_at_index(tensor<ShapeT, ET> &t, const IndexT &ind) {
  return t[ind];
}

template <class ET, class ShapeT, class ST, class... SizeTs>
void reserve_shape(tensor<ShapeT, ET> &t,
                   const tensor_shape<ST, SizeTs...> &shape) {
  assert(t.shape() == shape);
}

template <class FunT, class ET, class ShapeT, class... Ts>
void for_each_element(FunT &&fun, const tensor<ShapeT, ET> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 0; i < numel(t); i++) {
    fun(element_at_index(t, i), element_at_index(ts, i)...);
  }
}
template <class FunT, class ET, class ShapeT, class... Ts>
void for_each_element(FunT &&fun, tensor<ShapeT, ET> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 0; i < numel(t); i++) {
    fun(element_at_index(t, i), element_at_index(ts, i)...);
  }
}
template <class FunT, class ET, class ShapeT, class... Ts>
void for_each_nonzero_element(FunT &&fun, const tensor<ShapeT, ET> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 0; i < numel(t); i++) {
    decltype(auto) e = element_at_index(t, i);
    if (e) {
      fun(e, element_at_index(ts, i)...);
    }
  }
}
template <class FunT, class ET, class ShapeT, class... Ts>
void for_each_nonzero_element(FunT &&fun, tensor<ShapeT, ET> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 0; i < numel(t); i++) {
    decltype(auto) e = element_at_index(t, i);
    if (e) {
      fun(e, element_at_index(ts, i)...);
    }
  }
}
template <class FunT, class ET, class ShapeT, class... Ts>
bool for_each_element_if(FunT &&fun, const tensor<ShapeT, ET> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 0; i < numel(t); i++) {
    if (!fun(element_at_index(t, i), element_at_index(ts, i)...))
      return false;
  }
  return true;
}
template <class FunT, class ET, class ShapeT, class... Ts>
bool for_each_element_if(FunT &&fun, tensor<ShapeT, ET> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 0; i < numel(t); i++) {
    if (!fun(element_at_index(t, i), element_at_index(ts, i)...))
      return false;
  }
  return true;
}



template <class T, size_t N>
using vec_ = tensor<tensor_shape<size_t, const_size<N>>, T>;
using vec2 = vec_<double, 2>;
using vec3 = vec_<double, 3>;

template <class T> using vecx_ = tensor<tensor_shape<size_t, size_t>, T>;
using vecx = vecx_<double>;

template <class T, size_t M, size_t N>
using mat_ = tensor<tensor_shape<size_t, const_size<M>, const_size<N>>, T>;
using mat2 = mat_<double, 2, 2>;
using mat3 = mat_<double, 3, 3>;

template <class T, size_t M, size_t N, size_t L>
using cube_ =
    tensor<tensor_shape<size_t, const_size<M>, const_size<N>, const_size<L>>,
           T>;
using cube2 = cube_<double, 2, 2, 2>;
using cube3 = cube_<double, 3, 3, 3>;

// tensor_of_rank
namespace details {
template <class T, class SeqT> struct _make_tensor_of_rank_seq {
  using type = void;
};
template <class T, size_t... Is>
struct _make_tensor_of_rank_seq<T, const_ints<size_t, Is...>> {
  using type = tensor<tensor_shape<size_t, always_t<size_t, Is>...>, T>;
};
}
template <class T, size_t Rank>
using tensor_of_rank = typename details::_make_tensor_of_rank_seq<
    T, decltype(make_const_sequence(const_size<Rank>()))>::type;
}