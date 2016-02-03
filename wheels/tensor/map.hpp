#pragma once

#include "aligned_data.hpp"
#include "base.hpp"

namespace wheels {

// tensor_map_storage
template <class ShapeT, class ET, class PtrT, class T, bool StaticShape>
class tensor_map_storage;

template <class ShapeT, class ET, class PtrT, class T>
class tensor_map_storage<ShapeT, ET, PtrT, T, true>
    : public tensor_continuous_data_base<ShapeT, ET, T> {
public:
  using shape_type = ShapeT;
  using value_type = ET;

public:
  constexpr tensor_map_storage() : _ptr(nullptr) {}
  constexpr tensor_map_storage(const shape_type &, PtrT p) : _ptr(p) {}

  constexpr tensor_map_storage(const tensor_map_storage &) = default;
  tensor_map_storage(tensor_map_storage &&) = default;
  tensor_map_storage &operator=(const tensor_map_storage &) = default;
  tensor_map_storage &operator=(tensor_map_storage &&) = default;

  constexpr auto shape() const { return shape_type(); }
  constexpr PtrT ptr() const { return _ptr; }

private:
  PtrT _ptr;
};

template <class ShapeT, class ET, class PtrT, class T>
class tensor_map_storage<ShapeT, ET, PtrT, T, false>
    : public tensor_continuous_data_base<ShapeT, ET, T> {
public:
  using shape_type = ShapeT;
  using value_type = ET;

public:
  constexpr tensor_map_storage() : _ptr(nullptr) {}
  constexpr tensor_map_storage(const shape_type &s, PtrT p)
      : _shape(s), _ptr(p) {}

  constexpr tensor_map_storage(const tensor_map_storage &) = default;
  tensor_map_storage(tensor_map_storage &&) = default;
  tensor_map_storage &operator=(const tensor_map_storage &) = default;
  tensor_map_storage &operator=(tensor_map_storage &&) = default;

  constexpr const auto &shape() const { return _shape; }
  constexpr PtrT ptr() const { return _ptr; }

private:
  shape_type _shape;
  PtrT _ptr;
};

// tensor_map
template <class ShapeT, class ET, class PtrT = ET *>
class tensor_map
    : public tensor_map_storage<ShapeT, ET, PtrT, tensor_map<ShapeT, ET, PtrT>,
                                ShapeT::is_static> {
  using storage_t =
      tensor_map_storage<ShapeT, ET, PtrT, tensor_map<ShapeT, ET, PtrT>,
                         ShapeT::is_static>;

public:
  using shape_type = ShapeT;
  using value_type = ET;

public:
  constexpr tensor_map() : storage_t() {}
  constexpr tensor_map(PtrT ptr) : storage_t(shape_type(), ptr) {}
  constexpr tensor_map(const shape_type &s, PtrT ptr) : storage_t(s, ptr) {}

public:
  tensor_map(const tensor_map &) = default;
  tensor_map(tensor_map &&) = default;
  tensor_map &operator=(const tensor_map &) = default;
  tensor_map &operator=(tensor_map &&) = default;

  template <class AnotherT>
  constexpr tensor_map(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
  }
  template <class AnotherT>
  tensor_map &operator=(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
    return *this;
  }

public:
  constexpr const ET &at(size_t ind) const { return ptr()[ind]; }
  ET &at(size_t ind) { return ptr()[ind]; }
};

// ptr_of
template <class ET, class ShapeT, class PtrT>
constexpr auto ptr_of(const tensor_map<ShapeT, ET, PtrT> &t) {
  return t.ptr();
}
template <class ET, class ShapeT, class PtrT>
constexpr auto ptr_of(tensor_map<ShapeT, ET, PtrT> &t) {
  return t.ptr();
}

// shape_of
template <class ET, class ShapeT, class PtrT>
constexpr auto shape_of(const tensor_map<ShapeT, ET, PtrT> &t) {
  return t.shape();
}

// map
template <class E, class ST, class... SizeTs>
constexpr auto map(const tensor_shape<ST, SizeTs...> &shape, E *mem) {
  return tensor_map<tensor_shape<ST, SizeTs...>, E, E *>(shape, mem);
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
