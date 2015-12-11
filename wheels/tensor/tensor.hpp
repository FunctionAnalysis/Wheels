#pragma once

#include "base.hpp"

namespace wheels {

// dense tensor

// tensor_storage
constexpr struct _with_elements {
} with_elements;
constexpr struct _with_iterators {
} with_iterators;

template <class ShapeT, class ET, class T, bool StaticShape>
class tensor_storage;
namespace details {
template <class T, size_t N, size_t... Is>
constexpr auto _init_std_array_seq(const_ints<size_t, Is...>) {
  return std::array<T, N>{{(T)always<int, 0, const_index<Is>>::value...}};
}
template <class T, size_t N> constexpr auto _init_std_array() {
  return _init_std_array_seq<T, N>(make_const_sequence(const_size<N>()));
}
}
template <class ShapeT, class ET, class T>
class tensor_storage<ShapeT, ET, T, true> : public tensor_base<ShapeT, ET, T> {
public:
  using shape_type = ShapeT;
  using value_type = ET;

public:
  constexpr tensor_storage()
      : _data(details::_init_std_array<value_type,
                                       shape_type::static_magnitude>()) {}
  constexpr tensor_storage(const shape_type &s)
      : _data(details::_init_std_array<value_type,
                                       shape_type::static_magnitude>()) {
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
  constexpr const auto &data() const { return _data; }
  auto &data() { return _data; }

public:
  template <class ArcT> void serialize(ArcT &ar) { ar(_data); }
  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(_data);
  }
  template <class V> constexpr decltype(auto) fields(V &&visitor) const {
    return visitor(_data);
  }

private:
  std::array<value_type, shape_type::static_magnitude> _data;
};

template <class ShapeT, class ET, class T>
class tensor_storage<ShapeT, ET, T, false> : public tensor_base<ShapeT, ET, T> {
public:
  using shape_type = ShapeT;
  using value_type = ET;

public:
  constexpr tensor_storage() {}
  constexpr tensor_storage(const shape_type &shape)
      : _shape(shape), _data(shape.magnitude()) {}
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
  const auto &data() const { return _data; }
  auto &data() { return _data; }

public:
  template <class ArcT> void serialize(ArcT &ar) { ar(_shape, _data); }
  template <class V> decltype(auto) fields(V &&visitor) {
    return visitor(_shape, _data);
  }
  template <class V> constexpr decltype(auto) fields(V &&visitor) const {
    return visitor(_shape, _data);
  }

private:
  shape_type _shape;
  std::vector<value_type> _data;
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

template <class ShapeT, class ET>
class tensor
    : public tensor_storage<ShapeT, ET, tensor<ShapeT, ET>, ShapeT::is_static> {
  using base_t =
      tensor_storage<ShapeT, ET, tensor<ShapeT, ET>, ShapeT::is_static>;

public:
  using value_type = ET;
  using shape_type = ShapeT;

  constexpr tensor() : base_t() {}

  template <class... EleTs,
            class = std::enable_if_t<
                (ShapeT::dynamic_size_num == 0 &&
                 ::wheels::all(std::is_convertible<EleTs, ET>::value...))>>
  constexpr tensor(EleTs &&... eles)
      : base_t(ShapeT(), with_elements, forward<EleTs>(eles)...) {}

  template <class... EleTs, class = void,
            class = std::enable_if_t<
                (ShapeT::dynamic_size_num == 1 &&
                 ::wheels::all(std::is_convertible<EleTs, ET>::value...))>>
  constexpr tensor(EleTs &&... eles)
      : base_t(details::_make_shape_from_magnitude_seq<ShapeT>(
                   sizeof...(EleTs),
                   make_const_sequence(const_size<ShapeT::rank>())),
               with_elements, forward<EleTs>(eles)...) {}

  template <class... EleTs>
  constexpr tensor(const ShapeT &shape, const _with_elements &we,
                   EleTs &&... eles)
      : base_t(shape, we, forward<EleTs>(eles)...) {}

  constexpr tensor(const ShapeT &shape) : base_t(shape) {}

  template <class = std::enable_if_t<(ShapeT::dynamic_size_num == 0)>>
  constexpr tensor(std::initializer_list<value_type> ilist)
      : base_t(ShapeT(), with_iterators, ilist.begin(), ilist.end()) {}

  template <class = void,
            class = std::enable_if_t<(ShapeT::dynamic_size_num == 1)>>
  constexpr tensor(std::initializer_list<value_type> ilist)
      : base_t(
            details::_make_shape_from_magnitude_seq<ShapeT>(
                ilist.size(), make_const_sequence(const_size<ShapeT::rank>())),
            with_iterators, ilist.begin(), ilist.end()) {}

  constexpr tensor(const ShapeT &shape, std::initializer_list<value_type> ilist)
      : base_t(shape, with_iterators, ilist.begin(), ilist.end()) {}

  template <class IterT,
            class = std::enable_if_t<(ShapeT::dynamic_size_num == 0 &&
                                      is_iterator<IterT>::value)>>
  constexpr tensor(IterT begin, IterT end, yes * = nullptr)
      : base_t(ShapeT(), with_iterators, begin, end) {}

  template <class IterT,
            class = std::enable_if_t<(ShapeT::dynamic_size_num == 1 &&
                                      is_iterator<IterT>::value)>>
  constexpr tensor(IterT begin, IterT end, no * = nullptr)
      : base_t(details::_make_shape_from_magnitude_seq<ShapeT>(
                   std::distance(begin, end),
                   make_const_sequence(const_size<ShapeT::rank>())),
               with_iterators, begin, end) {}

  template <class IterT, class = std::enable_if_t<is_iterator<IterT>::value>>
  constexpr tensor(const ShapeT &shape, IterT begin, IterT end)
      : base_t(shape, with_iterators, begin, end) {}

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

public:
  constexpr decltype(auto) shape() const { return base_t::shape(); }
  template <class... SubTs>
  constexpr decltype(auto) operator()(const SubTs &... subs) const {
    static_assert(sizeof...(SubTs) == ShapeT::rank,
                  "invalid number of subscripts");
    return base_t::data()[sub2ind(shape(), subs...)];
  }
  template <class... SubTs> decltype(auto) operator()(const SubTs &... subs) {
    static_assert(sizeof...(SubTs) == ShapeT::rank,
                  "invalid number of subscripts");
    return base_t::data()[sub2ind(shape(), subs...)];
  }
  template <class IndexT>
  constexpr decltype(auto) operator[](const IndexT &ind) const {
    return base_t::data()[ind];
  }
  template <class IndexT> decltype(auto) operator[](const IndexT &ind) {
    return base_t::data()[ind];
  }
};

// necessary
template <class ET, class ShapeT>
constexpr auto shape_of(const tensor<ShapeT, ET> &t) {
  return t.shape();
}
template <class ET, class ShapeT, class... SubTs>
constexpr decltype(auto) element_at(const tensor<ShapeT, ET> &t,
                                    const SubTs &... subs) {
  return t(subs...);
}
template <class ET, class ShapeT, class... SubTs>
decltype(auto) element_at(tensor<ShapeT, ET> &t, const SubTs &... subs) {
  return t(subs...);
}

// auxiliary
template <class ET, class ShapeT, class IndexT>
constexpr decltype(auto) element_at_index(const tensor<ShapeT, ET> &t,
                                          const IndexT &ind) {
  return t[ind];
}
template <class ET, class ShapeT, class IndexT>
decltype(auto) element_at_index(tensor<ShapeT, ET> &t, const IndexT &ind) {
  return t[ind];
}

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

template <class FunT, class ET, class ShapeT, class... Ts>
void for_each_element(FunT &&fun, const tensor<ShapeT, ET> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 0; i < numel(t); i++) {
    fun(element_at_index(t, i), element_at_index(ts, i)...);
  }
}
template <class FunT, class ET, class ShapeT, class... Ts>
void for_each_element(FunT &&fun, tensor<ShapeT, ET> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 0; i < numel(t); i++) {
    fun(element_at_index(t, i), element_at_index(ts, i)...);
  }
}
template <class FunT, class ET, class ShapeT, class... Ts>
void for_each_nonzero_element(FunT &&fun, const tensor<ShapeT, ET> &t,
                              Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 0; i < numel(t); i++) {
    decltype(auto) e = element_at_index(t, i);
    if (e) {
      fun(e, element_at_index(ts, i)...);
    }
  }
}
template <class FunT, class ET, class ShapeT, class... Ts>
void for_each_nonzero_element(FunT &&fun, tensor<ShapeT, ET> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 0; i < numel(t); i++) {
    decltype(auto) e = element_at_index(t, i);
    if (e) {
      fun(e, element_at_index(ts, i)...);
    }
  }
}
template <class FunT, class ET, class ShapeT, class... Ts>
bool for_each_element_if(FunT &&fun, const tensor<ShapeT, ET> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 0; i < numel(t); i++) {
    if (!fun(element_at_index(t, i), element_at_index(ts, i)...))
      return false;
  }
  return true;
}
template <class FunT, class ET, class ShapeT, class... Ts>
bool for_each_element_if(FunT &&fun, tensor<ShapeT, ET> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 0; i < numel(t); i++) {
    if (!fun(element_at_index(t, i), element_at_index(ts, i)...))
      return false;
  }
  return true;
}

template <class T, size_t N>
using vec_ = tensor<tensor_shape<size_t, const_size<N>>, T>;
using vec2 = vec_<double, 2>;
using vec3 = vec_<double, 3>;

template <class T> using vecx_ = tensor<tensor_shape<size_t, size_t>, T>;
using vecx = vecx_<double>;

template <class T, size_t M, size_t N>
using mat_ = tensor<tensor_shape<size_t, const_size<M>, const_size<N>>, T>;
using mat2 = mat_<double, 2, 2>;
using mat3 = mat_<double, 3, 3>;

template <class T>
using matx_ = tensor<tensor_shape<size_t, size_t, size_t>, T>;
using matx = matx_<double>;

template <class T, size_t M, size_t N, size_t L>
using cube_ =
    tensor<tensor_shape<size_t, const_size<M>, const_size<N>, const_size<L>>,
           T>;
using cube2 = cube_<double, 2, 2, 2>;
using cube3 = cube_<double, 3, 3, 3>;

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