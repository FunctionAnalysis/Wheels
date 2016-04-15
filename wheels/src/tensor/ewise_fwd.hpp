#pragma once

#include "base.hpp"

namespace wheels {

// ewise_base
template <class EleT, class ShapeT, class T>
class ewise_base : public tensor_base<EleT, ShapeT, T> {
public:
  template <class FunT> constexpr auto transform(FunT &&fun) const &;
  template <class FunT> auto transform(FunT &&fun) &;
  template <class FunT> constexpr auto transform(FunT &&fun) &&;

  template <class TargetEleT> constexpr auto cast() const &;
  template <class TargetEleT> constexpr auto cast() &&;
};

namespace details {
template <class EleT, class ShapeT, class T, class TT>
constexpr auto _ewise(const tensor_base<EleT, ShapeT, T> &, TT &&host);
template <class EleT, class ShapeT, class T, class TT>
constexpr decltype(auto) _ewise(const ewise_base<EleT, ShapeT, T> &,
                                TT &&host);
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

}