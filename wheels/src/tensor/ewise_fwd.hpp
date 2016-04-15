#pragma once

#include "base.hpp"

namespace wheels {

// ewise_base
template <class EleT, class ShapeT, class OpT, class T> class ewise_base;

namespace details {
template <class EleT, class ShapeT, class T, class TT>
constexpr auto _ewise(const tensor_base<EleT, ShapeT, T> &, TT &&host);
template <class EleT, class ShapeT, class OpT, class T, class TT>
constexpr decltype(auto) _ewise(const ewise_base<EleT, ShapeT, OpT, T> &,
                                TT &&host);
}

// ewise
template <class T>
constexpr auto ewise(T &&t)
    -> decltype(details::_ewise(t, std::forward<T>(t))) {
  return details::_ewise(t, std::forward<T>(t));
}


}