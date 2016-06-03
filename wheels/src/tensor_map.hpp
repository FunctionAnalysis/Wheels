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
namespace details {
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
  using info_t = details::_raw_array_info<E[N]>;
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
