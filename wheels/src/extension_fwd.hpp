#pragma once

#include "tensor_base_fwd.hpp"

namespace wheels {

template <class ExtentionT, class EleT, class ShapeT, class T>
class tensor_extension_base;

template <class ExtensionT, class EleT, class ShapeT, class T>
class tensor_extension_wrapper;

namespace detail {
template <class ExtensionT, class EleT, class ShapeT, class T, class TT>
constexpr auto _extend(const tensor_base<EleT, ShapeT, T> &, TT &&host);
}

// extend
template <class ExtensionT, class T>
constexpr auto extend(T &&host)
    -> decltype(detail::_extend<ExtensionT>(host, std::forward<T>(host))) {
  return detail::_extend<ExtensionT>(host, std::forward<T>(host));
}

}
