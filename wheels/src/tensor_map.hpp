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

#include "aligned.hpp"
#include "tensor_base.hpp"

#include "storage.hpp"
#include "tensor_view_base.hpp"

#include "tensor_map_fwd.hpp"

namespace wheels {
// tensor_map
template <class ET, class ShapeT>
class tensor_map : public tensor_view_base<std::decay_t<ET>, ShapeT,
                                           tensor_map<ET, ShapeT>, true> {
  using _base_t =
      tensor_view_base<std::decay_t<ET>, ShapeT, tensor_map<ET, ShapeT>, true>;

public:
  using shape_type = ShapeT;
  using value_type = std::decay_t<ET>;

public:
  constexpr tensor_map() : _storage() {}
  constexpr tensor_map(ET *ptr) : _storage(shape_type(), ptr) {}
  constexpr tensor_map(const shape_type &s, ET *ptr) : _storage(s, ptr) {}

public:
  tensor_map(const tensor_map &) = delete;
  tensor_map(tensor_map &&) = default;
  tensor_map &operator=(const tensor_map &) = delete;
  tensor_map &operator=(tensor_map &&) = default;

  constexpr const ET *ptr() const { return _storage.data(); }
  ET *ptr() { return _storage.data(); }
  constexpr decltype(auto) shape() const { return _storage.shape(); }
  void reshape(const shape_type &s) { _storage.reshape(s); }

  using _base_t::operator=;
  using _base_t::operator+=;
  using _base_t::operator-=;
  using _base_t::operator*=;
  using _base_t::operator/=;

private:
  map_storage<ET, shape_type> _storage;
};

// ptr_of
template <class ET, class ShapeT>
constexpr auto ptr_of(const tensor_map<ET, ShapeT> &t) {
  return t.ptr();
}
template <class ET, class ShapeT>
constexpr auto ptr_of(tensor_map<ET, ShapeT> &t) {
  return t.ptr();
}

// shape_of
template <class ET, class ShapeT>
constexpr decltype(auto) shape_of(const tensor_map<ET, ShapeT> &t) {
  return t.shape();
}

// map
template <class E, class ST, class... SizeTs>
constexpr auto map(const tensor_shape<ST, SizeTs...> &shape, E *mem) {
  return tensor_map<E, tensor_shape<ST, SizeTs...>>(shape, mem);
}

// from raw array
namespace detail {
template <class E> struct _raw_array_info {
  using ele_t = E;
  static auto shape() { return tensor_shape<size_t>(); }
};
template <class E, size_t N> struct _raw_array_info<E[N]> {
  using ele_t = typename _raw_array_info<E>::ele_t;
  static auto shape() {
    return ::wheels::cat(const_size<N>(), _raw_array_info<E>::shape());
  }
};
}
template <class E, size_t N> constexpr auto map(E (&arr)[N]) {
  using info_t = detail::_raw_array_info<E[N]>;
  return map(info_t::shape(), (typename info_t::ele_t *)arr);
}

// from raw string
namespace literals {
inline auto operator"" _ts(const char *str, size_t s) {
  return map(make_shape(s), str);
}
inline auto operator"" _ts(const wchar_t *str, size_t s) {
  return map(make_shape(s), str);
}
inline auto operator"" _ts(const char16_t *str, size_t s) {
  return map(make_shape(s), str);
}
inline auto operator"" _ts(const char32_t *str, size_t s) {
  return map(make_shape(s), str);
}
}
}
