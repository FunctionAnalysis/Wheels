#pragma once

#include "tensor_base_fwd.hpp"

namespace wheels {
template <class ET, class ShapeT, class T> class reshape_view;

namespace detail {
template <class ET, class OldShapeT, class T, class TT, class ShapeT>
constexpr auto _reshape(const tensor_base<ET, OldShapeT, T> &, TT &&t,
                        const ShapeT &s);
template <class ET, class OldShapeT, class T, class TT, class ShapeT>
constexpr auto _reshape(const reshape_view<ET, OldShapeT, T> &, TT &&t,
                        const ShapeT &s);
}
// reshape
template <class T, class ST, class... SizeTs>
constexpr auto reshape(T &&t, const tensor_shape<ST, SizeTs...> &s)
    -> decltype(detail::_reshape(t, std::forward<T>(t), s)) {
  return detail::_reshape(t, std::forward<T>(t), s);
}

namespace detail {
template <class ET, class ST, class... SizeTs, class T, class TT, class K,
          K Times>
constexpr auto _promote(const tensor_base<ET, tensor_shape<ST, SizeTs...>, T> &,
                        TT &&t, const const_ints<K, Times> &);
template <class ET, class ST, class... SizeTs, class T, class TT, class K,
          K Times>
constexpr auto _promote(const const_ints<K, Times> &,
                        const tensor_base<ET, tensor_shape<ST, SizeTs...>, T> &,
                        TT &&t);
}
// promote
template <class T, class K, K Times>
constexpr auto promote(T &&t, const const_ints<K, Times> &rank)
    -> decltype(detail::_promote(t, std::forward<T>(t), rank)) {
  return detail::_promote(t, std::forward<T>(t), rank);
}
template <class T, class K, K Times>
constexpr auto promote(const const_ints<K, Times> &rank, T &&t)
    -> decltype(detail::_promote(rank, t, std::forward<T>(t))) {
  return detail::_promote(rank, t, std::forward<T>(t));
}
}
