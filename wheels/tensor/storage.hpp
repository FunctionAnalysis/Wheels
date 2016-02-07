#pragma once

#include "shape.hpp"

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

constexpr struct _with_elements {
} with_elements;
constexpr struct _with_iterators {
} with_iterators;

template <class T, class ShapeT, bool ShapeIsStatic = ShapeT::is_static>
class storage;

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
      : _data{{(value_type)forward<EleTs>(eles)...}} {}
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
// initialize_n
template <class T>
inline std::enable_if_t<std::is_scalar<T>::value> _initialize_n(T *data,
                                                                size_t n) {
  std::fill_n(data, n, T(0));
}
template <class T>
inline std::enable_if_t<!std::is_scalar<T>::value> _initialize_n(T *data,
                                                                 size_t n) {
  for (size_t i = 0; i < n; i++) {
    void *ptr = (void *)(data + i);
    ::new (ptr) T();
  }
}

// initialize_n_by_move
template <class T>
inline std::enable_if_t<std::is_scalar<T>::value>
_initialize_n_by_move(T *data, size_t n, T *src) {
  std::copy_n(src, n, data);
}
template <class T>
inline std::enable_if_t<!std::is_scalar<T>::value>
_initialize_n_by_move(T *data, size_t n, T *src) {
  for (size_t i = 0; i < n; i++) {
    void *ptr = (void *)(data + i);
    ::new (ptr) T(std::move(src[i]));
  }
}

// initialize_by_elements
template <class T> inline void _initialize_by_elements(T *data) {}
template <class T, class EleT, class... EleTs>
inline std::enable_if_t<std::is_scalar<T>::value>
_initialize_by_elements(T *data, EleT &&ele, EleTs &&... eles) {
  *data = (T)ele;
  _initialize_by_elements(data + 1, std::forward<EleTs>(eles)...);
}
template <class T, class EleT, class... EleTs>
inline std::enable_if_t<!std::is_scalar<T>::value>
_initialize_by_elements(T *data, EleT &&ele, EleTs &&... eles) {
  ::new ((void *)data) T(std::forward<EleT>(ele));
  _initialize_by_elements(data + 1, std::forward<EleTs>(eles)...);
}
}

// dynamic shaped storage
template <class T, class ShapeT> class storage<T, ShapeT, false> {
  static_assert(is_tensor_shape<ShapeT>::value,
                "ShapeT must be a tensor_shape");

public:
  using value_type = T;
  using shape_type = ShapeT;

  static constexpr size_t _initial_cap = 2;
  storage() : _shape(), _capacity(_initial_cap) {
    _data = new value_type[_capacity];
    details::_initialize_n(_data, _capacity);
  }
  explicit storage(const shape_type &s) : _shape(s) {
    _capacity = _shape.magnitude();
    _data = new value_type[_capacity];
    details::_initialize_n(_data, _capacity);
  }
  storage(const shape_type &s, const value_type &e) : _shape(s) {
    _capacity = _shape.magnitude();
    _data = new value_type[_capacity];
    std::fill_n(_data, _capacity, e);
  }
  template <class... EleTs>
  storage(const shape_type &s, _with_elements, EleTs &&... eles) : _shape(s) {
    _capacity = _shape.magnitude();
    _data = new value_type[_capacity];
    details::_initialize_by_elements(_data, std::forward<EleTs>(eles)...);
  }
  template <class IterT>
  storage(const shape_type &s, _with_iterators, IterT begin, IterT end)
      : _shape(s) {
    _capacity = _shape.magnitude();
    _data = new value_type[_capacity];
    std::copy(begin, end, _data);
  }

  storage(const storage &st) : _shape(st._shape) {
    _capacity = _shape.magnitude();
    _data = new value_type[_capacity];
    std::copy_n(st._data, _capacity, _data);
  }
  storage(storage &&st)
      : _shape(st._shape), _capacity(st._capacity), _data(st._data) {
    st._shape = shape_type();
    st._data = nullptr;
  }

  ~storage() {
    delete[] _data;
    _data = nullptr;
  }

  storage &operator=(const storage &st) {
    _shape = st._shape;
    const auto nmag = _shape.magnitude();
    if (_capacity < nmag) {
      _capacity = nmag;
      value_type *ndata = new value_type[_capacity];
      delete[] _data;
      _data = ndata;
    }
    std::copy_n(st._data, _capacity, _data);
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
  }

  void reshape(const shape_type &nshape) {
    const auto mag = _shape.magnitude();
    const auto nmag = nshape.magnitude();
    if (_capacity < nmag) {
      _capacity = nmag;
      value_type *ndata = new value_type[_capacity];
      details::_initialize_n_by_move(ndata, mag, _data);
      delete[] _data;
      _data = ndata;
    }
    if (mag < nmag) {
      details::_initialize_n(_data + mag, nmag - mag);
    }
    _shape = nshape;
  }

  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(
        as_container(make_range(_data, _data + _shape.magnitude()), visitor));
  }

private:
  shape_type _shape;
  size_t _capacity;
  value_type *_data;
};

// serialize
template <class T, class ShapeT, bool S, class ArcT>
void save(ArcT &ar, const storage<T, ShapeT, S> &st) {
  details::_save_shape(ar, st.shape());
  ar(cereal::binary_data(st.data(), sizeof(T) * st.shape().magnitude()));
}
template <class T, class ShapeT, bool S, class ArcT>
void load(ArcT &ar, storage<T, ShapeT, S> &st) {
  ShapeT s;
  details::_load_shape(ar, s);
  st.reshape(s);
  ar(cereal::binary_data(st.data(), sizeof(T) * s.magnitude()));
}
}
