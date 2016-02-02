#pragma once

#include "aligned_data.hpp"
#include "base.hpp"

namespace wheels {

// dense tensor

constexpr struct _with_elements {
} with_elements;
constexpr struct _with_iterators {
} with_iterators;

// tensor_storage
template <class ShapeT, class ET, class T, bool StaticShape>
class tensor_storage;

namespace details {
template <class T, size_t N, size_t... Is>
constexpr auto _init_std_array_seq(const T &init, const_ints<size_t, Is...>) {
  return std::array<T, N>{{always_f<T>(init)(const_index<Is>())...}};
}
template <class T, size_t N> constexpr auto _init_std_array(const T &init) {
  return _init_std_array_seq<T, N>(init, make_const_sequence(const_size<N>()));
}
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

// static tensor
template <class ShapeT, class ET, class T>
class tensor_storage<ShapeT, ET, T, true>
    : public tensor_continuous_data_base<ShapeT, ET, T> {
public:
  using shape_type = ShapeT;
  using value_type = ET;

public:
  constexpr tensor_storage()
      : _data(
            details::_init_std_array<value_type, shape_type::static_magnitude>(
                0)) {}
  constexpr tensor_storage(const shape_type &s)
      : _data(
            details::_init_std_array<value_type, shape_type::static_magnitude>(
                0)) {
    assert(s.magnitude() == _data.size());
  }
  constexpr tensor_storage(const shape_type &s, const value_type &v)
      : _data(
            details::_init_std_array<value_type, shape_type::static_magnitude>(
                v)) {
    assert(s.magnitude() == _data.size());
  }
  template <class... EleTs>
  constexpr tensor_storage(const shape_type &shape, const _with_elements &,
                           EleTs &&... eles)
      : _data{{(value_type)forward<EleTs>(eles)...}} {}
  template <class IterT>
  tensor_storage(const shape_type &shape, const _with_iterators &, IterT begin,
                 IterT end) {
    std::copy(begin, end, _data.begin());
  }

  constexpr tensor_storage(const tensor_storage &) = default;
  tensor_storage(tensor_storage &&) = default;
  tensor_storage &operator=(const tensor_storage &) = default;
  tensor_storage &operator=(tensor_storage &&) = default;

  constexpr auto shape() const { return shape_type(); }
  constexpr const auto &container() const { return _data; }
  auto &container() { return _data; }

public:
  constexpr decltype(auto) at(size_t ind) const { return _data[ind]; }
  decltype(auto) at(size_t ind) { return _data[ind]; }

public:
  template <class ArcT> void save(ArcT &ar) const {
    details::_save_shape(ar, shape());
    ar(cereal::make_size_tag(_data.size()));
    ar(cereal::binary_data(_data.data(), sizeof(value_type) * _data.size()));
  }
  template <class ArcT> void load(ArcT &ar) {
    auto s = shape();
    details::_load_shape(ar, s);
    size_t n;
    ar(cereal::make_size_tag(n));
    assert(n == _data.size());
    ar(cereal::binary_data(_data.data(), sizeof(value_type) * _data.size()));
  }

  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(_data);
  }

private:
  std::array<value_type, shape_type::static_magnitude> _data;
};

// dynamic tensor
template <class ShapeT, class ET, class T>
class tensor_storage<ShapeT, ET, T, false>
    : public tensor_continuous_data_base<ShapeT, ET, T> {
public:
  using shape_type = ShapeT;
  using value_type = ET;

public:
  constexpr tensor_storage() {}
  constexpr tensor_storage(const shape_type &shape)
      : _shape(shape), _data(shape.magnitude()) {}
  constexpr tensor_storage(const shape_type &shape, const value_type &v)
      : _shape(shape), _data(shape.magnitude(), v) {}
  template <class... EleTs>
  constexpr tensor_storage(const shape_type &shape, const _with_elements &,
                           EleTs &&... eles)
      : _shape(shape), _data({(value_type)forward<EleTs>(eles)...}) {
    _data.resize(_shape.magnitude());
  }
  template <class IterT>
  tensor_storage(const shape_type &shape, const _with_iterators &, IterT begin,
                 IterT end)
      : _shape(shape), _data(begin, end) {
    _data.resize(_shape.magnitude());
  }

  constexpr tensor_storage(const tensor_storage &) = default;
  tensor_storage(tensor_storage &&) = default;
  tensor_storage &operator=(const tensor_storage &) = default;
  tensor_storage &operator=(tensor_storage &&) = default;

  constexpr const auto &shape() const { return _shape; }
  template <class ShapeT2> void set_shape(const ShapeT2 &s) {
    _shape = s;
    _data.resize(_shape.magnitude());
  }
  const auto &container() const { return _data; }
  auto &container() { return _data; }

public:
  constexpr decltype(auto) at(size_t ind) const { return _data[ind]; }
  decltype(auto) at(size_t ind) { return _data[ind]; }

public:
  template <class ArcT> void save(ArcT &ar) const {
    details::_save_shape(ar, _shape);
    ar(cereal::make_size_tag(_data.size()));
    ar(cereal::binary_data(_data.data(), sizeof(value_type) * _data.size()));
  }
  template <class ArcT> void load(ArcT &ar) {
    details::_load_shape(ar, _shape);
    size_t n;
    ar(cereal::make_size_tag(n));
    _data.resize(n);
    ar(cereal::binary_data(_data.data(), sizeof(value_type) * _data.size()));
  }

  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(_data);
  }

private:
  shape_type _shape;
  std::vector<value_type> _data;
};

// dynamic boolean tensor
template <class ShapeT, class T>
class tensor_storage<ShapeT, bool, T, false>
    : public tensor_continuous_data_base<ShapeT, bool, T> {
public:
  using shape_type = ShapeT;
  using value_type = bool;
  using stored_value_type = typename details::_uint_of<sizeof(bool)>::type;

public:
  constexpr tensor_storage() {}
  constexpr tensor_storage(const shape_type &shape)
      : _shape(shape), _data(shape.magnitude()) {}
  constexpr tensor_storage(const shape_type &shape, const value_type &v)
      : _shape(shape), _data(shape.magnitude(), (stored_value_type)v) {}
  template <class... EleTs>
  constexpr tensor_storage(const shape_type &shape, const _with_elements &,
                           EleTs &&... eles)
      : _shape(shape), _data({(stored_value_type)forward<EleTs>(eles)...}) {
    _data.resize(_shape.magnitude());
  }
  template <class IterT>
  tensor_storage(const shape_type &shape, const _with_iterators &, IterT begin,
                 IterT end)
      : _shape(shape) {
    _data.resize(_shape.magnitude());
    size_t i = 0;
    for (auto it = begin; it != end; ++it) {
      _data[i++] = (stored_value_type)*it;
    }
  }

  constexpr tensor_storage(const tensor_storage &) = default;
  tensor_storage(tensor_storage &&) = default;
  tensor_storage &operator=(const tensor_storage &) = default;
  tensor_storage &operator=(tensor_storage &&) = default;

  constexpr const auto &shape() const { return _shape; }
  template <class ShapeT2> void set_shape(const ShapeT2 &s) {
    _shape = s;
    _data.resize(_shape.magnitude());
  }
  const auto &container() const { return _data; }
  auto &container() { return _data; }

public:
  constexpr bool at(size_t ind) const { return _data[ind]; }
  bool &at(size_t ind) { return (bool &)(_data[ind]); }

public:
  template <class ArcT> void save(ArcT &ar) const {
    details::_save_shape(ar, _shape);
    ar(cereal::make_size_tag(_data.size()));
    ar(cereal::binary_data(_data.data(),
                           sizeof(stored_value_type) * _data.size()));
  }
  template <class ArcT> void load(ArcT &ar) {
    details::_load_shape(ar, _shape);
    size_t n;
    ar(cereal::make_size_tag(n));
    _data.resize(n);
    ar(cereal::binary_data(_data.data(),
                           sizeof(stored_value_type) * _data.size()));
  }

  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(_data);
  }

private:
  shape_type _shape;
  std::vector<stored_value_type> _data;
};

namespace details {
template <class ShapeT, size_t... Is>
constexpr ShapeT _make_shape_from_magnitude_seq(size_t magnitude,
                                                const_ints<size_t, Is...>) {
  static_assert(ShapeT::dynamic_size_num == 1,
                "ShapeT::dynamic_size_num should be 1 here");
  static_assert(ShapeT::last_dynamic_dim >= 0,
                "ShapeT::last_dynamic_dim is not valid");
  return ShapeT(conditional(const_bool<Is == ShapeT::last_dynamic_dim>(),
                            magnitude / ShapeT::static_magnitude,
                            std::ignore)...);
}
}

// tensor
template <class ShapeT, class ET>
class tensor
    : public tensor_storage<ShapeT, ET, tensor<ShapeT, ET>, ShapeT::is_static> {
  using storage_t =
      tensor_storage<ShapeT, ET, tensor<ShapeT, ET>, ShapeT::is_static>;

public:
  using value_type = ET;
  using shape_type = ShapeT;

  // tensor()
  constexpr tensor() : storage_t() {}

  // tensor(shape)
  constexpr tensor(const ShapeT &shape) : storage_t(shape) {}

  // tensor(shape, e)
  constexpr tensor(const ShapeT &shape, const value_type &v)
      : storage_t(shape, v) {}

  // tensor(shape, with_elements, e1, e2, e3 ...)
  template <class... EleTs>
  constexpr tensor(const ShapeT &shape, const _with_elements &we,
                   EleTs &&... eles)
      : storage_t(shape, we, forward<EleTs>(eles)...) {}

  // tensor({e1, e2, e3 ...})
  template <class EleT,
            class = std::enable_if_t<(ShapeT::dynamic_size_num == 0)>>
  constexpr tensor(std::initializer_list<EleT> ilist)
      : storage_t(ShapeT(), with_iterators, ilist.begin(), ilist.end()) {}
  template <class EleT, class = void,
            class = std::enable_if_t<(ShapeT::dynamic_size_num == 1)>>
  constexpr tensor(std::initializer_list<EleT> ilist)
      : storage_t(
            details::_make_shape_from_magnitude_seq<ShapeT>(
                ilist.size(), make_const_sequence(const_size<ShapeT::rank>())),
            with_iterators, ilist.begin(), ilist.end()) {}

  // tensor(shape, {e1, e2, e3 ...})
  constexpr tensor(const ShapeT &shape, std::initializer_list<value_type> ilist)
      : storage_t(shape, with_iterators, ilist.begin(), ilist.end()) {}

  // tensor(begin, end)
  template <class IterT,
            class = std::enable_if_t<(ShapeT::dynamic_size_num == 0 &&
                                      is_iterator<IterT>::value)>>
  constexpr tensor(IterT begin, IterT end, yes * = nullptr)
      : storage_t(ShapeT(), with_iterators, begin, end) {}

  template <class IterT,
            class = std::enable_if_t<(ShapeT::dynamic_size_num == 1 &&
                                      is_iterator<IterT>::value)>>
  constexpr tensor(IterT begin, IterT end, no * = nullptr)
      : storage_t(details::_make_shape_from_magnitude_seq<ShapeT>(
                      std::distance(begin, end),
                      make_const_sequence(const_size<ShapeT::rank>())),
                  with_iterators, begin, end) {}

  // tensor(shape, begin, end)
  template <class IterT, class = std::enable_if_t<is_iterator<IterT>::value>>
  constexpr tensor(const ShapeT &shape, IterT begin, IterT end)
      : storage_t(shape, with_iterators, begin, end) {}

  // tensor(e1, e2, e3 ...)
  template <class... EleTs,
            bool B = (ShapeT::dynamic_size_num == 0 &&
                      ::wheels::all(std::is_convertible<EleTs, ET>::value...)),
            class = std::enable_if_t<B>>
  constexpr tensor(EleTs &&... eles)
      : storage_t(ShapeT(), with_elements, forward<EleTs>(eles)...) {}

  template <class... EleTs, class = void,
            bool B = (ShapeT::dynamic_size_num == 1 &&
                      ::wheels::all(std::is_convertible<EleTs, ET>::value...)),
            class = std::enable_if_t<B>>
  constexpr tensor(EleTs &&... eles)
      : storage_t(details::_make_shape_from_magnitude_seq<ShapeT>(
                      sizeof...(EleTs),
                      make_const_sequence(const_size<ShapeT::rank>())),
                  with_elements, forward<EleTs>(eles)...) {}

  tensor(const tensor &) = default;
  tensor(tensor &&) = default;
  tensor &operator=(const tensor &) = default;
  tensor &operator=(tensor &&) = default;

  template <class AnotherT>
  constexpr tensor(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
  }
  template <class AnotherT>
  tensor &operator=(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
    return *this;
  }
};

// shape_of
template <class ET, class ShapeT>
constexpr auto shape_of(const tensor<ShapeT, ET> &t) {
  return t.shape();
}

// ptr_of
template <class ET, class ShapeT>
constexpr const ET *ptr_of(const tensor<ShapeT, ET> &t) {
  return (const ET *)(t.container().data());
}
template <class ET, class ShapeT> ET *ptr_of(tensor<ShapeT, ET> &t) {
  return (ET *)(t.container().data());
}

// reserve_shape
template <class ET, class ShapeT, class T, class ST, class... SizeTs>
void reserve_shape(tensor_storage<ShapeT, ET, T, true> &t,
                   const tensor_shape<ST, SizeTs...> &shape) {
  assert(t.shape() == shape);
}
template <class ET, class ShapeT, class T, class ST, class... SizeTs>
void reserve_shape(tensor_storage<ShapeT, ET, T, false> &t,
                   const tensor_shape<ST, SizeTs...> &shape) {
  t.set_shape(shape);
}

// vec_
template <class T, size_t N>
using vec_ = tensor<tensor_shape<size_t, const_size<N>>, T>;
using vec2 = vec_<double, 2>;
using vec3 = vec_<double, 3>;

// vecx_
template <class T> using vecx_ = tensor<tensor_shape<size_t, size_t>, T>;
using vecx = vecx_<double>;

// mat_
template <class T, size_t M, size_t N>
using mat_ = tensor<tensor_shape<size_t, const_size<M>, const_size<N>>, T>;
using mat2 = mat_<double, 2, 2>;
using mat3 = mat_<double, 3, 3>;

// matx_
template <class T>
using matx_ = tensor<tensor_shape<size_t, size_t, size_t>, T>;
using matx = matx_<double>;

// cube_
template <class T, size_t M, size_t N, size_t L>
using cube_ =
    tensor<tensor_shape<size_t, const_size<M>, const_size<N>, const_size<L>>,
           T>;
using cube2 = cube_<double, 2, 2, 2>;
using cube3 = cube_<double, 3, 3, 3>;

// cubex_
template <class T>
using cubex_ = tensor<tensor_shape<size_t, size_t, size_t, size_t>, T>;
using cubex = matx_<double>;

// tensor_of_rank
namespace details {
template <class T, class SeqT> struct _make_tensor_of_rank_seq {
  using type = void;
};
template <class T, size_t... Is>
struct _make_tensor_of_rank_seq<T, const_ints<size_t, Is...>> {
  using type = tensor<tensor_shape<size_t, always_t<size_t, Is>...>, T>;
};
}
template <class T, size_t Rank>
using tensor_of_rank = typename details::_make_tensor_of_rank_seq<
    T, decltype(make_const_sequence(const_size<Rank>()))>::type;
}