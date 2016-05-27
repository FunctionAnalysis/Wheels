#pragma once

namespace wheels {
// types
template <class... Ts> struct types;

// type_t
#define type_t(t) typename decltype(t)::type

template <class... Ts> constexpr auto type_of(Ts &&... t);
}