#pragma once

#include "shape.hpp"

#include "storage_fwd.hpp"

namespace wheels {

namespace details {
template <class ArcT, class ST, class... SS>
void _save_shape(ArcT &ar, const tensor_shape<ST, SS...> &s) {
  ar(sizeof...(SS));
  tensor_shape<size_t, always2_t<size_t, SS>...> stdshape = s;
  ar(stdshape);
}
template <class ArcT, class ST, class... SS>
void _load_shape(ArcT &ar, tensor_shape<ST, SS...> &s) {
  size_t rank;
  ar(rank);
  assert(rank == sizeof...(SS));
  tensor_shape<size_t, always2_t<size_t, SS>...> stdshape;
  ar(stdshape);
  s = stdshape;
}
}

namespace details {
// init_std_array
template <class T, size_t N, size_t... Is>
constexpr auto _init_std_array_seq(const T &init, const_ints<size_t, Is...>) {
  return std::array<T, N>{{always_f<T>(init)(const_index<Is>())...}};
}
template <class T, size_t N> constexpr auto _init_std_array(const T &init) {
  return _init_std_array_seq<T, N>(init, make_const_sequence(const_size<N>()));
}
}

// static shaped storage
template <class T, class ShapeT> class storage<T, ShapeT, true> {
  static_assert(is_tensor_shape<ShapeT>::value,
                "ShapeT must be a tensor_shape");

public:
  using value_type = T;
  using shape_type = ShapeT;

public:
  constexpr storage()
      : _data(details::_init_std_array<T, shape_type::static_magnitude>(
            value_type())) {}
  constexpr explicit storage(const shape_type &)
      : _data(details::_init_std_array<T, shape_type::static_magnitude>(
            value_type())) {}
  constexpr storage(const shape_type &, const value_type &e)
      : _data(details::_init_std_array<T, shape_type::static_magnitude>(e)) {}
  template <class... EleTs>
  constexpr storage(const shape_type &, const _with_elements &,
                    EleTs &&... eles)
      : _data{{(value_type)std::forward<EleTs>(eles)...}} {}
  template <class IterT>
  storage(const shape_type &, const _with_iterators &, IterT begin, IterT end)
      : _data{{value_type()}} {
    std::copy(begin, end, _data.begin());
  }

  constexpr storage(const storage &) = default;
  constexpr storage(storage &&) = default;

  storage &operator=(const storage &) = default;
  storage &operator=(storage &&) = default;

  constexpr shape_type shape() const { return shape_type(); }
  constexpr const value_type *data() const { return _data.data(); }
  value_type *data() { return _data.data(); }

  void reshape(const shape_type &nshape) { assert(nshape == shape()); }

  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(_data);
  }

private:
  std::array<value_type, shape_type::static_magnitude> _data;
};

namespace details {
//// initialize_n
// template <class T>
// inline std::enable_if_t<std::is_scalar<T>::value> _initialize_n(T *data,
//                                                                size_t n) {
//  std::fill_n(data, n, T(0));
//}
// template <class T>
// inline std::enable_if_t<!std::is_scalar<T>::value> _initialize_n(T *data,
//                                                                 size_t n) {
//  for (size_t i = 0; i < n; i++) {
//    void *ptr = (void *)(data + i);
//    ::new (ptr) T();
//  }
//}
//
//// initialize_n_by_move
// template <class T>
// inline std::enable_if_t<std::is_scalar<T>::value>
//_initialize_n_by_move(T *data, size_t n, T *src) {
//  std::copy_n(src, n, data);
//}
// template <class T>
// inline std::enable_if_t<!std::is_scalar<T>::value>
//_initialize_n_by_move(T *data, size_t n, T *src) {
//  for (size_t i = 0; i < n; i++) {
//    void *ptr = (void *)(data + i);
//    ::new (ptr) T(std::move(src[i]));
//  }
//}

// _construct_n
template <class T, class AllocT, class... ArgTs>
inline void _construct_n(T *data, AllocT &alloc, size_t n, ArgTs &... args) {
  for (size_t i = 0; i < n; i++) {
    alloc.construct(data + i, args...);
  }
}

// _construct_each_by
template <class T, class AllocT>
inline void _construct_each_by(T *data, AllocT &alloc) {}
template <class T, class AllocT, class EleT, class... EleTs>
inline void _construct_each_by(T *data, AllocT &alloc, EleT &&ele,
                               EleTs &&... eles) {
  alloc.construct(data, std::forward<EleT>(ele));
  _construct_each_by(data + 1, alloc, std::forward<EleTs>(eles)...);
}
}

// dynamic shaped storage
template <class T, class ShapeT> class storage<T, ShapeT, false> {
  static_assert(is_tensor_shape<ShapeT>::value,
                "ShapeT must be a tensor_shape");

public:
  using value_type = T;
  using shape_type = ShapeT;

  static constexpr size_t _initial_cap = 1;
  storage() : _shape(), _capacity(_initial_cap) {
    _data = _alloc.allocate(_capacity);
    details::_construct_n(_data, _alloc, _capacity);
  }
  explicit storage(const shape_type &s) : _shape(s) {
    _capacity = _shape.magnitude();
    _data = _alloc.allocate(_capacity);
    details::_construct_n(_data, _alloc, _capacity);
  }
  storage(const shape_type &s, const value_type &e) : _shape(s) {
    _capacity = _shape.magnitude();
    _data = _alloc.allocate(_capacity);
    details::_construct_n(_data, _alloc, _capacity, e);
  }
  template <class... EleTs>
  storage(const shape_type &s, _with_elements, EleTs &&... eles) : _shape(s) {
    _capacity = _shape.magnitude();
    _data = _alloc.allocate(_capacity);
    details::_construct_each_by(_data, _alloc, std::forward<EleTs>(eles)...);
    for (size_t i = sizeof...(EleTs); i < _capacity; i++) {
      _alloc.construct(_data + i);
    }
  }
  template <class IterT>
  storage(const shape_type &s, _with_iterators, IterT begin, IterT end)
      : _shape(s) {
    _capacity = _shape.magnitude();
    _data = _alloc.allocate(_capacity);
    size_t i = 0;
    for (; i < _capacity && begin != end; i++) {
      _alloc.construct(_data + i, *begin);
      ++begin;
    }
    for (; i < _capacity; i++) {
      _alloc.construct(_data + i);
    }
  }

  storage(const storage &st) : _shape(st._shape) {
    _capacity = _shape.magnitude();
    _data = _alloc.allocate(_capacity);
    for (size_t i = 0; i < _capacity; i++) {
      _alloc.construct(_data + i, st._data[i]);
    }
  }
  storage(storage &&st)
      : _shape(st._shape), _capacity(st._capacity), _data(st._data),
        _alloc(st._alloc) {
    st._shape = shape_type();
    st._data = nullptr;
    st._capacity = 0;
  }

  ~storage() {
    if (_data) {
      for (size_t i = 0; i < _capacity; i++) {
        _alloc.destroy(_data + i);
      }
      _alloc.deallocate(_data, _capacity);
      _capacity = 0;
      _data = nullptr;
    }
  }

  storage &operator=(const storage &st) {
    _shape = st._shape;
    const auto nmag = _shape.magnitude();
    if (_capacity < nmag) {
      value_type *ndata = _alloc.allocate(nmag);
      for (size_t i = 0; i < nmag; i++) {
        _alloc.construct(ndata + i, st._data[i]);
      }
      for (size_t i = 0; i < _capacity; i++) {
        _alloc.destroy(_data + i);
      }
      _alloc.deallocate(_data, _capacity);
      _capacity = nmag;
      _data = ndata;
    } else {
      for (size_t i = 0; i < nmag; i++) {
        _data[i] = st._data[i];
      }
    }
    return *this;
  }
  storage &operator=(storage &&st) {
    swap(st);
    return *this;
  }

  constexpr const shape_type &shape() const { return _shape; }
  constexpr const value_type *data() const { return _data; }
  value_type *data() { return _data; }
  constexpr size_t capacity() const { return _capacity; }

  void swap(storage &st) {
    std::swap(_shape, st._shape);
    std::swap(_capacity, st._capacity);
    std::swap(_data, st._data);
    std::swap(_alloc, st._alloc);
  }

  void reshape(const shape_type &nshape) {
    const auto mag = _shape.magnitude();
    const auto nmag = nshape.magnitude();
    if (_capacity < nmag) { // allocate new
      value_type *ndata = _alloc.allocate(nmag);
      for (size_t i = 0; i < _capacity; i++) {
        _alloc.construct(ndata + i, std::move(_data[i]));
      }
      for (size_t i = _capacity; i < nmag; i++) {
        _alloc.construct(ndata + i);
      }
      for (size_t i = 0; i < _capacity; i++) {
        _alloc.destroy(_data + i);
      }
      _alloc.deallocate(_data, _capacity);
      _capacity = nmag;
      _data = ndata;
    }
    if (mag < nmag) {
      for (size_t i = mag; i < nmag; i++) {
        _data[i] = value_type();
      }
    }
    _shape = nshape;
  }

  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(
        as_container(make_interval(_alloc.address(_data[0]),
                                   _alloc.address(_data[_shape.magnitude()])),
                     visitor));
  }

private:
  shape_type _shape;
  size_t _capacity;
  value_type *_data;
  std::allocator<value_type> _alloc;
};
}
