#pragma once

#include "base_fwd.hpp"

namespace wheels {

// ewise_base
template <class EleT, class ShapeT, class T>
class ewise_base : public tensor_base<EleT, ShapeT, T> {
public:
  using value_type = EleT;
  using shape_type = ShapeT;

  template <class FunT> constexpr auto transform(FunT &&fun) const &;
  template <class FunT> auto transform(FunT &&fun) &;
  template <class FunT> constexpr auto transform(FunT &&fun) &&;

  template <class TargetEleT> constexpr auto cast() const &;
  template <class TargetEleT> constexpr auto cast() &&;
};

template <class EleT, class ShapeT, class T> class ewise_wrapper;

namespace details {
template <class EleT, class ShapeT, class T, class TT>
constexpr auto _ewise(const tensor_base<EleT, ShapeT, T> &, TT &&host);
template <class EleT, class ShapeT, class T, class TT>
constexpr TT &&_ewise(const ewise_base<EleT, ShapeT, T> &, TT &&host);
}

// ewise
template <class T>
constexpr auto ewise(T &&t)
    -> decltype(details::_ewise(t, std::forward<T>(t))) {
  return details::_ewise(t, std::forward<T>(t));
}

// ewise_op_result
template <class EleT, class ShapeT, class OpT, class InputT, class... InputTs>
class ewise_op_result;

namespace details {
template <class FirstTT, class... TTs, size_t... Is, class FirstEleT,
          class... EleTs, class FirstShapeT, class... ShapeTs, class FirstT,
          class... Ts>
constexpr auto _as_tuple_seq(std::tuple<FirstTT, TTs...> &&ts,
                             const const_index<Is...> &,
                             const ewise_base<FirstEleT, FirstShapeT, FirstT> &,
                             const ewise_base<EleTs, ShapeTs, Ts> &...);
}

// as_tuple
template <class FirstT, class... Ts>
constexpr auto as_tuple(FirstT &&t, Ts &&ts...)
    -> decltype(details::_as_tuple_seq(
        std::forward_as_tuple(std::forward<FirstT>(t), std::forward<Ts>(ts)...),
        make_const_sequence_for<Ts...>(), t, ts...)) {
  return details::_as_tuple_seq(
      std::forward_as_tuple(std::forward<FirstT>(t), std::forward<Ts>(ts)...),
      make_const_sequence_for<Ts...>(), t, ts...);
}
}