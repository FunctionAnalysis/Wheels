#pragma once

namespace wheels {
namespace category {

// object
template <class T> struct object;

// other
template <class T> struct other;

// identify
template <class T> constexpr decltype(auto) identify(const T &t);
}

// eval
template <class T> constexpr auto eval(const T & v);
}
