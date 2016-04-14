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
}
