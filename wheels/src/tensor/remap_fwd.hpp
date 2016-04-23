#pragma once

#include "../core/const_ints.hpp"
#include "../core/types.hpp"

#include "base_fwd.hpp"

namespace wheels {

// interpolate_method
enum interpolate_method_enum { round_interpolate, linear_interpolate };
template <interpolate_method_enum IM>
using interpolate_method = const_ints<interpolate_method_enum, IM>;

template <class ET, class ShapeT, class T, class MapFunT,
          interpolate_method_enum IPMethod>
class remap_result;

// remap
template <class ToST, class... ToSizeTs, class ET, class ShapeT, class T,
          class MapFunT, class ET2 = ET,
          interpolate_method_enum IPMethod = linear_interpolate>
constexpr auto remap(const tensor_base<ET, ShapeT, T> &t,
                     const tensor_shape<ToST, ToSizeTs...> &toshape,
                     MapFunT mapfun, ET2 &&outlier = types<ET2>::zero(),
                     const interpolate_method<IPMethod> & = {});

template <class ToST, class... ToSizeTs, class ET, class ShapeT, class T,
          class MapFunT, class ET2 = ET,
          interpolate_method_enum IPMethod = linear_interpolate>
constexpr auto remap(tensor_base<ET, ShapeT, T> &&t,
                     const tensor_shape<ToST, ToSizeTs...> &toshape,
                     MapFunT mapfun, ET2 &&outlier = types<ET2>::zero(),
                     interpolate_method<IPMethod> = {});
}
