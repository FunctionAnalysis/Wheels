#pragma once

#include "base_fwd.hpp"

namespace wheels {

template <class ET, class ShapeT, class T, size_t... Inds> class permute_result;

namespace details {
template <class ET, class ShapeT, class T, class TT, class... IndexTs>
constexpr decltype(auto) _permute(const tensor_base<ET, ShapeT, T> &, TT &&t,
                                  const IndexTs &...);
template <class ET, class ShapeT, class T, size_t... Inds, class TT,
          class... IndexTs>
constexpr decltype(auto)
_permute(const permute_result<ET, ShapeT, T, Inds...> &, TT &&t,
         const IndexTs &...);
}

template <class T, class... IndexTs>
constexpr auto permute(T &&t, const IndexTs &... inds)
    -> decltype(details::_permute(t, std::forward<T>(t), inds...)) {
  return details::_permute(t, std::forward<T>(t), inds...);
}

// transpose
namespace details {
template <class ST, class MT, class NT, class ET, class T, class TT>
constexpr auto _transpose(const tensor_base<ET, tensor_shape<ST, MT, NT>, T> &,
                          TT &&t)
    -> decltype(permute(std::forward<TT>(t), const_index<1>(), const_index<0>())) {
  return permute(std::forward<TT>(t), const_index<1>(), const_index<0>());
}
}
template <class T>
constexpr auto transpose(T &&t)
    -> decltype(details::_transpose(t, std::forward<T>(t))) {
  return details::_transpose(t, std::forward<T>(t));
}
}
