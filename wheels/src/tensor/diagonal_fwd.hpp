#pragma once

#include "shape_fwd.hpp"
#include "base_fwd.hpp"

namespace wheels {

// make_diag_result
template <class ET, class ShapeT, class T> class make_diag_result;

// diag_view
template <class ET, class ShapeT, class T> class diag_view;

// make_diag
namespace details {
template <class ET, class ShapeT, class T, class TT, class NewShapeT>
constexpr auto _make_diag(const tensor_base<ET, ShapeT, T> &, TT &&t,
                          const NewShapeT &nshape);
}
template <class T, class ST, class... SizeTs>
constexpr auto make_diag(T &&t, const tensor_shape<ST, SizeTs...> &ns)
    -> decltype(details::_make_diag(t, std::forward<T>(t), ns)) {
  return details::_make_diag(t, std::forward<T>(t), ns);
}
template <class T>
constexpr auto make_diag(T &&t)
    -> decltype(details::_make_diag(t, std::forward<T>(t),
                                    make_shape(t.numel(), t.numel()))) {
  return details::_make_diag(t, std::forward<T>(t),
                             make_shape(t.numel(), t.numel()));
}

// eye
template <class ET = double, class ST, class... SizeTs>
constexpr auto eye(const tensor_shape<ST, SizeTs...> &s);
template <class ET = double, class MT, class NT>
constexpr auto eye(const MT &m, const NT &n);
template <class ET = double, class NT,
          class = std::enable_if_t<!is_tensor_shape<NT>::value>>
constexpr auto eye(const NT &n);

// diag
namespace details {
template <class ET, class ShapeT, class T, class TT>
constexpr auto _diag(const tensor_base<ET, ShapeT, T> &, TT &&t);
template <class ET, class ShapeT, class T, class TT>
constexpr decltype(auto) _diag(const make_diag_result<ET, ShapeT, T> &, TT &&t);
}
template <class T>
constexpr auto diag(T &&t) -> decltype(details::_diag(t, std::forward<T>(t))) {
  return details::_diag(t, std::forward<T>(t));
}
}
