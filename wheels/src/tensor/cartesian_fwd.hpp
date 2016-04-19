#pragma once

#include "base_fwd.hpp"

namespace wheels {

// meshgrid
template <class ET, class ShapeT, size_t Axis> class meshgrid_result;

template <class ET = size_t, class ST, class... SizeTs, class K, K Axis>
constexpr auto meshgrid(const tensor_shape<ST, SizeTs...> &s,
                        const const_ints<K, Axis> &);

template <class ET = size_t, class ST, class... SizeTs>
constexpr auto meshgrid(const tensor_shape<ST, SizeTs...> &s);

// coordinate
template <class ET, class ShapeT> class coordinate_result;

template <class ET = size_t, class ST, class... SizeTs>
constexpr auto coordinate(const tensor_shape<ST, SizeTs...> &s);

// cart_prod
template <class TupleT, class ShapeT, class... Ts> class cart_prod_result;

namespace details {
template <class... Ts, class... TTs> constexpr auto _cart_prod(TTs &&... tts);
}
template <class... TTs>
constexpr auto cart_prod(TTs &&... ts)
    -> decltype(details::_cart_prod(std::forward<TTs>(ts)...)) {
  return details::_cart_prod(std::forward<TTs>(ts)...);
}
}
