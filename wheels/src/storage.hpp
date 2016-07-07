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

#include "shape.hpp"

#include "storage_fwd.hpp"

namespace wheels {
namespace detail {
// init_std_array
template <class T, size_t N, size_t... Is>
constexpr auto _init_std_array_seq(const T &init, const_ints<size_t, Is...>) {
  return std::array<T, N>{{always_f<T>(init)(const_index<Is>())...}};
}
}
template <class T, size_t N> constexpr auto init_std_array(const T &init) {
  return detail::_init_std_array_seq<T, N>(
      init, make_const_sequence(const_size<N>()));
}

// uninitialized_default_fill_n (refering the same name function in msvc stl)
namespace detail {
// for normal types initialization
template <class IterT, class DiffT, class AllocT>
void _uninitialized_default_fill_n(IterT first, DiffT count, AllocT &alloc,
                                   no) {
  IterT next = first;
  wheels_try {
    for (; 0 < count; --count, (void)++first)
      alloc.construct(first);
  }
  wheels_catch_all {
    for (; next != first; ++next)
      alloc.destroy(next);
    wheels_rethrow;
  }
}
// for scalars initialization
template <class IterT, class DiffT>
void _uninitialized_default_fill_n(
    IterT first, DiffT count,
    std::allocator<typename std::iterator_traits<IterT>::value_type> &alloc,
    yes) {
  memset(first, 0,
         count * sizeof(typename std::iterator_traits<IterT>::value_type));
}
}
template <class IterT, class DiffT, class AllocT>
inline void uninitialized_default_fill_n(IterT first, DiffT count,
                                         AllocT &alloc) {
  typedef typename std::iterator_traits<IterT>::value_type T;
  detail::_uninitialized_default_fill_n(
      first, count, alloc, const_bool < std::is_pointer<IterT>::value &&
                               std::is_scalar<T>::value &&
                               !std::is_volatile<T>::value &&
                               !std::is_member_pointer<T>::value > ());
}

template <class IterT, class DiffT, class AllocT, class ValT>
inline void uninitialized_default_fill_n(IterT first, DiffT count,
                                         AllocT &alloc, const ValT &val) {
  IterT next = first;
  wheels_try {
    for (; 0 < count; --count, (void)++first)
      alloc.construct(first, val);
  }
  wheels_catch_all {
    for (; next != first; ++next)
      alloc.destroy(next);
    wheels_rethrow;
  }
}

// uninitialized_default_fill_args
template <class IterT, class AllocT>
inline void uninitialized_default_fill_args(IterT first, AllocT &alloc) {}
template <class IterT, class AllocT, class EleT, class... EleTs>
inline void uninitialized_default_fill_args(IterT first, AllocT &alloc,
                                            EleT &&ele, EleTs &&... eles) {
  alloc.construct(first, std::forward<EleT>(ele));
  uninitialized_default_fill_args(++first, alloc, std::forward<EleTs>(eles)...);
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
      : _data(init_std_array<T, shape_type::static_magnitude>(value_type())) {}
  constexpr explicit storage(const shape_type &)
      : _data(init_std_array<T, shape_type::static_magnitude>(value_type())) {}
  constexpr storage(const shape_type &, const value_type &e)
      : _data(init_std_array<T, shape_type::static_magnitude>(e)) {}
  template <class... EleTs>
  constexpr storage(const shape_type &, const _with_elements &,
                    EleTs &&... eles)
      : _data{{(value_type)std::forward<EleTs>(eles)...}} {
    static_assert(sizeof...(EleTs) == shape_type::static_magnitude,
                  "number of elements mismatch with the size of storage");
  }
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

private:
  std::array<value_type, shape_type::static_magnitude> _data;
};

// static shaped map_storage
template <class T, class ShapeT> class map_storage<T, ShapeT, true> {
  static_assert(is_tensor_shape<ShapeT>::value,
                "ShapeT must be a tensor_shape");

public:
  using value_type = T;
  using shape_type = ShapeT;

public:
  constexpr map_storage() : _ptr(nullptr) {}
  constexpr explicit map_storage(const shape_type &, value_type *ptr)
      : _ptr(ptr) {}

  map_storage(const map_storage &) = delete;
  map_storage &operator=(const map_storage &) = delete;
  map_storage(map_storage &&) = default;
  map_storage &operator=(map_storage &&) = default;

  constexpr shape_type shape() const { return shape_type(); }
  constexpr const value_type *data() const { return _ptr; }
  value_type *data() { return _ptr; }

  void reshape(const shape_type &nshape) { assert(nshape == shape()); }

private:
  value_type *_ptr;
};

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
    uninitialized_default_fill_n(_data, _capacity, _alloc);
  }
  explicit storage(const shape_type &s) : _shape(s) {
    _capacity = _shape.magnitude();
    _data = _alloc.allocate(_capacity);
    uninitialized_default_fill_n(_data, _capacity, _alloc);
  }
  storage(const shape_type &s, const value_type &e) : _shape(s) {
    _capacity = _shape.magnitude();
    _data = _alloc.allocate(_capacity);
    uninitialized_default_fill_n(_data, _capacity, _alloc, e);
  }
  template <class... EleTs>
  storage(const shape_type &s, _with_elements, EleTs &&... eles) : _shape(s) {
    _capacity = _shape.magnitude();
    _data = _alloc.allocate(_capacity);
    uninitialized_default_fill_args(_data, _alloc,
                                    std::forward<EleTs>(eles)...);
    if (sizeof...(EleTs) < _capacity) {
      uninitialized_default_fill_n(_data + sizeof...(EleTs),
                                   _capacity - sizeof...(EleTs), _alloc);
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

private:
  shape_type _shape;
  size_t _capacity;
  value_type *_data;
  std::allocator<value_type> _alloc;
};

template <class T, class ShapeT> class map_storage<T, ShapeT, false> {
  static_assert(is_tensor_shape<ShapeT>::value,
                "ShapeT must be a tensor_shape");

public:
  using value_type = T;
  using shape_type = ShapeT;

  constexpr map_storage() : _ptr(nullptr) {}
  constexpr explicit map_storage(const shape_type &s, value_type *ptr)
      : _shape(s), _ptr(ptr) {}

  map_storage(const map_storage &) = delete;
  map_storage &operator=(const map_storage &) = delete;
  map_storage(map_storage &&) = default;
  map_storage &operator=(map_storage &&) = default;

  constexpr const shape_type &shape() const { return _shape; }
  constexpr const value_type *data() const { return _ptr; }
  value_type *data() { return _ptr; }

  void reshape(const shape_type &nshape) {
    assert(nshape.magnitude() == _shape.magnitude());
    _shape = nshape;
  }

private:
  shape_type _shape;
  value_type *_ptr;
};
}
