#pragma once

#include "const_ints.hpp"
#include "types.hpp"

#include "tensor_base_fwd.hpp"

namespace wheels {

// interpolate_method
enum interpolate_method_enum { round_interpolate, linear_interpolate };
template <interpolate_method_enum IM>
using interpolate_method = const_ints<interpolate_method_enum, IM>;

template <class ET, class ShapeT, class T, class MapFunT,
          interpolate_method_enum IPMethod>
class remap_result;

// remap
namespace detail {
// w. outlier
template <class ToST, class... ToSizeTs, class ET, class ShapeT, class T,
          class TT, class MapFunT, class ET2, interpolate_method_enum IPMethod>
constexpr auto _remap(const tensor_base<ET, ShapeT, T> &, TT &&t,
                      const tensor_shape<ToST, ToSizeTs...> &toshape,
                      MapFunT mapfun, ET2 &&outlier,
                      const interpolate_method<IPMethod> &);
// w.o outlier
template <class ToST, class... ToSizeTs, class ET, class ShapeT, class T,
          class TT, class MapFunT, interpolate_method_enum IPMethod>
constexpr auto _remap(const tensor_base<ET, ShapeT, T> &, TT &&t,
                      const tensor_shape<ToST, ToSizeTs...> &toshape,
                      MapFunT mapfun, const interpolate_method<IPMethod> &);
}
// w. outlier
template <class ToST, class... ToSizeTs, class T, class MapFunT, class ET,
          interpolate_method_enum IPMethod = linear_interpolate>
constexpr auto remap(T &&t, const tensor_shape<ToST, ToSizeTs...> &toshape,
                     MapFunT mapfun, ET &&outlier,
                     const interpolate_method<IPMethod> &m = {})
    -> decltype(detail::_remap(t, std::forward<T>(t), toshape, mapfun,
                                std::forward<ET>(outlier), m)) {
  return detail::_remap(t, std::forward<T>(t), toshape, mapfun,
                         std::forward<ET>(outlier), m);
}
// w.o outlier
template <class ToST, class... ToSizeTs, class T, class MapFunT,
          interpolate_method_enum IPMethod = linear_interpolate>
constexpr auto remap(T &&t, const tensor_shape<ToST, ToSizeTs...> &toshape,
                     MapFunT mapfun, const interpolate_method<IPMethod> &m = {})
    -> decltype(detail::_remap(t, std::forward<T>(t), toshape, mapfun, m)) {
  return detail::_remap(t, std::forward<T>(t), toshape, mapfun, m);
}

namespace detail {
template <class FromShapeT, class ToShapeT> struct _resample_map_functor;
template <class FromShapeT, class ToShapeT>
_resample_map_functor<FromShapeT, ToShapeT>
_make_resample_map_functor(const FromShapeT &from_shape,
                           const ToShapeT &to_shape);
}

// resample
template <class ToST, class... ToSizeTs, class T,
          interpolate_method_enum IPMethod = linear_interpolate>
constexpr auto
resample(T &&t, const tensor_shape<ToST, ToSizeTs...> &toshape,
         const interpolate_method<IPMethod> & = interpolate_method<IPMethod>())
    -> decltype(detail::_remap(t, std::forward<T>(t), toshape,
                                detail::_make_resample_map_functor(t.shape(),
                                                                    toshape),
                                interpolate_method<IPMethod>())) {
  return detail::_remap(
      t, std::forward<T>(t), toshape,
      detail::_make_resample_map_functor(t.shape(), toshape),
      interpolate_method<IPMethod>());
}
}
