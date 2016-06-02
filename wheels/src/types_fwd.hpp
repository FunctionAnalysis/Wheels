#pragma once

namespace wheels {
// types
template <class... Ts> struct types;

// type_t
#define type_t(t) typename decltype(t)::type

template <class... Ts> constexpr auto type_of(Ts &&... t);

enum cast_type_enum {
  by_static,
  by_dynamic,
  by_reinterpret,
  by_construct,
  by_c_style
};
template <cast_type_enum cast_type, class T, class K> constexpr T cast(K &&v);
}