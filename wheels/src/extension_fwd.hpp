#pragma once

#include "tensor_base_fwd.hpp"

namespace wheels {

template <class ExtentionT, class EleT, class ShapeT, class T>
class tensor_extension_base;

template <class ExtensionT, class EleT, class ShapeT, class T>
class tensor_extension_wrapper;

namespace details {
template <class ExtensionT, class EleT, class ShapeT, class T, class TT>
constexpr auto _extend(const tensor_base<EleT, ShapeT, T> &, TT &&host);
template <class ExtensionT, class EleT, class ShapeT, class T, class TT>
constexpr TT &&
_extend(const tensor_extension_base<ExtensionT, EleT, ShapeT, T> &, TT &&host);
}

// extend
template <class ExtensionT, class T>
constexpr auto extend(T &&host)
    -> decltype(details::_extend<ExtensionT>(host, std::forward<T>(host))) {
  return details::_extend<ExtensionT>(host, std::forward<T>(host));
}

}
