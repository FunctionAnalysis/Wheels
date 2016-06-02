#pragma once

namespace wheels {
// object_base
template <class T> struct object_base;

// proxy_base
template <class T> struct proxy_base;

// what
template <class T> constexpr decltype(auto) what(const T &);
}
