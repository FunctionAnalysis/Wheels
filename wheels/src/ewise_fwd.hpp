#pragma once

#include "const_expr_fwd.hpp"
#include "overloads_fwd.hpp"

#include "const_ints.hpp"

#include "extension_fwd.hpp"
#include "tensor_base_fwd.hpp"

namespace wheels {

// explicitly declare as ewise op for disambiguity
struct extension_tag_ewise {};

template <class EleT, class ShapeT, class T>
class tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>
    : public tensor_base<EleT, ShapeT, T> {
public:
  using value_type = EleT;
  using shape_type = ShapeT;

  template <class AnotherT> constexpr auto equals(AnotherT &&t) const &;
  template <class AnotherT> auto equals(AnotherT &&t) &;
  template <class AnotherT> auto equals(AnotherT &&t) &&;

  template <class AnotherT> constexpr auto not_equals(AnotherT &&t) const &;
  template <class AnotherT> auto not_equals(AnotherT &&t) &;
  template <class AnotherT> auto not_equals(AnotherT &&t) &&;

  template <class FunT> constexpr auto transform(FunT &&fun) const &;
  template <class FunT> auto transform(FunT &&fun) &;
  template <class FunT> auto transform(FunT &&fun) &&;

  template <class TargetEleT> constexpr auto cast() const &;
  template <class TargetEleT> auto cast() &&;
};

// ewise_base
template <class EleT, class ShapeT, class T>
using ewise_base = tensor_extension_base<extension_tag_ewise, EleT, ShapeT, T>;

template <class EleT, class ShapeT, class T>
using ewise_wrapper =
    tensor_extension_wrapper<extension_tag_ewise, EleT, ShapeT, T>;

// ewise
template <class T>
constexpr auto ewise(T &&t)
    -> decltype(extend<extension_tag_ewise>(std::forward<T>(t))) {
  return extend<extension_tag_ewise>(std::forward<T>(t));
}

// explicitly treat as a scalar in ewise ops
struct extension_tag_scalarize {};

template <class EleT, class ShapeT, class T>
class tensor_extension_base<extension_tag_scalarize, EleT, ShapeT, T>
    : public tensor_base<EleT, ShapeT, T> {
public:
  using value_type = EleT;
  using shape_type = ShapeT;
};

// scalarize_wrapper
template <class EleT, class ShapeT, class T>
using scalarize_wrapper =
    tensor_extension_wrapper<extension_tag_scalarize, EleT, ShapeT, T>;

// scalarize
template <class T>
constexpr auto scalarize(T &&t)
    -> decltype(extend<extension_tag_scalarize>(std::forward<T>(t))) {
  return extend<extension_tag_scalarize>(std::forward<T>(t));
}

// ewise_op_result
template <class EleT, class ShapeT, class OpT, class InputT, class... InputTs>
class ewise_op_result;

namespace details {
template <class FirstTT, class... TTs, size_t... Is, class FirstEleT,
          class... EleTs, class FirstShapeT, class... ShapeTs, class FirstT,
          class... Ts>
constexpr auto _as_tuple_seq(std::tuple<FirstTT, TTs...> &&ts,
                             const const_ints<size_t, Is...> &,
                             const ewise_base<FirstEleT, FirstShapeT, FirstT> &,
                             const ewise_base<EleTs, ShapeTs, Ts> &...);
}

// most ewise binray ops apply on two tensors (except certain ops like below)
struct binary_op_eq;
struct binary_op_neq;
struct binary_op_mul;
namespace details {
template <class T> struct _op_naturally_ewise : yes {};
template <> struct _op_naturally_ewise<binary_op_eq> : no {};
template <> struct _op_naturally_ewise<binary_op_neq> : no {};
template <> struct _op_naturally_ewise<binary_op_mul> : no {};
}
template <class OpT,
          class = std::enable_if_t<details::_op_naturally_ewise<OpT>::value>,
          class EleT, class ShapeT, class T, class... EleTs, class... ShapeTs,
          class... Ts>
constexpr auto overload_as(const func_base<OpT> &op,
                           const tensor_base<EleT, ShapeT, T> &t,
                           const tensor_base<EleTs, ShapeTs, Ts> &... ts);

// all ewise binary ops apply on ewised tensor
// t1.ewise() == t2, t1.ewise() * t2
template <class OpT, class EleT, class ShapeT, class T, class... EleTs,
          class... ShapeTs, class... Ts>
constexpr auto overload_as(const func_base<OpT> &op,
                           const ewise_wrapper<EleT, ShapeT, T> &t,
                           const tensor_base<EleTs, ShapeTs, Ts> &... ts);

// tensor vs scalar
template <class OpT, class EleT1, class ShapeT1, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const tensor_base<EleT1, ShapeT1, T1> &,
                           const category::other<T2> &);

template <class OpT, class EleT1, class ShapeT1, class EleT2, class ShapeT2,
          class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const tensor_base<EleT1, ShapeT1, T1> &,
                           const scalarize_wrapper<EleT2, ShapeT2, T2> &);

// scalar vs tensor
template <class OpT, class T1, class EleT2, class ShapeT2, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const category::other<T1> &,
                           const tensor_base<EleT2, ShapeT2, T2> &);

template <class OpT, class EleT1, class ShapeT1, class EleT2, class ShapeT2,
          class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const scalarize_wrapper<EleT1, ShapeT1, T1> &,
                           const tensor_base<EleT2, ShapeT2, T2> &);

// tensor vs const_expr
template <class OpT, class EleT1, class ShapeT1, class T1, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const tensor_base<EleT1, ShapeT1, T1> &,
                           const const_expr_base<T2> &);

// const_expr vs tensor
template <class OpT, class T1, class EleT2, class ShapeT2, class T2>
constexpr auto overload_as(const func_base<OpT> &op,
                           const const_expr_base<T1> &,
                           const tensor_base<EleT2, ShapeT2, T2> &);

// as_tuple
template <class FirstT, class... Ts>
constexpr auto as_tuple(FirstT &&t, Ts &&... ts)
    -> decltype(details::_as_tuple_seq(
        std::forward_as_tuple(std::forward<FirstT>(t), std::forward<Ts>(ts)...),
        make_const_sequence_for<Ts...>(), t, ts...)) {
  return details::_as_tuple_seq(
      std::forward_as_tuple(std::forward<FirstT>(t), std::forward<Ts>(ts)...),
      make_const_sequence_for<Ts...>(), t, ts...);
}
}