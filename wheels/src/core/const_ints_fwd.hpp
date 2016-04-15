#pragma once

#include <cstdint>

namespace wheels {

// const_ints
template <class T, T... Vals> struct const_ints;

template <bool Val> using const_bool = const_ints<bool, Val>;
template <int Val> using const_int = const_ints<int, Val>;
template <size_t Val> using const_size = const_ints<size_t, Val>;
template <size_t Val> using const_index = const_ints<size_t, Val>;

using yes = const_bool<true>;
using no = const_bool<false>;

namespace literals {
// ""_c
template <char... Cs> constexpr auto operator"" _c();

// ""_uc
template <char... Cs> constexpr auto operator"" _uc();

// ""_sizec
template <char... Cs> constexpr auto operator"" _sizec();

// ""_indexc
template <char... Cs> constexpr auto operator"" _indexc();

// ""_int8c
template <char... Cs> constexpr auto operator"" _int8c();

// ""_int16c
template <char... Cs> constexpr auto operator"" _int16c();

// ""_int32c
template <char... Cs> constexpr auto operator"" _int32c();

// ""_int64c
template <char... Cs> constexpr auto operator"" _int64c();

// ""_uint8c
template <char... Cs> constexpr auto operator"" _uint8c();

// ""_uint16c
template <char... Cs> constexpr auto operator"" _uint16c();

// ""_uint32c
template <char... Cs> constexpr auto operator"" _uint32c();

// ""_uint64c
template <char... Cs> constexpr auto operator"" _uint64c();
}
}
