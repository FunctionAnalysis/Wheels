/* * *
 * The MIT License (MIT)
 * 
 * Copyright (c) 2016 Hao Yang (yangh2007@gmail.com)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * * */

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

// cat2
template <class T, T... Val1s, class K, K... Val2s>
constexpr auto cat2(const const_ints<T, Val1s...> &,
                    const const_ints<K, Val2s...> &);

// conditional
template <class T, T Val, class ThenT, class ElseT>
constexpr std::enable_if_t<Val, ThenT &&>
conditional(const const_ints<T, Val> &, ThenT &&thenv, ElseT &&elsev);
template <class T, T Val, class ThenT, class ElseT, bool _B = Val>
constexpr std::enable_if_t<!_B, ElseT &&>
conditional(const const_ints<T, Val> &, ThenT &&thenv, ElseT &&elsev);

// make_const_sequence
template <class T, T Size>
constexpr auto make_const_sequence(const const_ints<T, Size> &);

// make_const_sequence_for
template <class... Ts> constexpr auto make_const_sequence_for();

// repeat
template <class T, T Val, class K, K Times>
constexpr auto repeat(const const_ints<T, Val> &v,
                      const const_ints<K, Times> &times);

// count
template <class T, T... S, class K, K V>
constexpr auto count(const const_ints<T, S...> &seq, const const_ints<K, V> &v);

// find_first_of
template <class T, T S, T... Ss, class K, K V>
constexpr auto find_first_of(const const_ints<T, S, Ss...> &seq,
                             const const_ints<K, V> &v);

// for_each
template <class T, class FunT>
inline void for_each(const const_ints<T> &, FunT fun);
template <class T, T S, T... Ss, class FunT>
inline void for_each(const const_ints<T, S, Ss...> &, FunT fun);
}
