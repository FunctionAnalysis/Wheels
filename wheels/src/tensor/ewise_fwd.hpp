#pragma once

#include "base_fwd.hpp"
#include "extension_fwd.hpp"

namespace wheels {

struct extension_tag_ewise;

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

// ewise
template <class T>
constexpr auto ewise(T &&t)
    -> decltype(extend<extension_tag_ewise>(std::forward<T>(t))) {
  return extend<extension_tag_ewise>(std::forward<T>(t));
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