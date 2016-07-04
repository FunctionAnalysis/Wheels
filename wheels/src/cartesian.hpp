#pragma once

#include "types.hpp"

#include "tensor_base.hpp"
#include "tensor.hpp"

#include "cartesian_fwd.hpp"

namespace wheels {

/// meshgrid
// meshgrid_result
template <class ET, class ShapeT, size_t Axis>
class meshgrid_result
    : public tensor_base<ET, ShapeT, meshgrid_result<ET, ShapeT, Axis>> {
  static_assert(Axis < ShapeT::rank, "Axis overflow");

public:
  constexpr explicit meshgrid_result(const ShapeT &s) : _shape(s) {}
  constexpr const ShapeT &shape() const { return _shape; }

private:
  ShapeT _shape;
};

// shape_of
template <class ET, class ShapeT, size_t Axis>
constexpr decltype(auto) shape_of(const meshgrid_result<ET, ShapeT, Axis> &t) {
  return t.shape();
}

// element_at
template <class ET, class ShapeT, size_t Axis, class... SubTs>
constexpr ET element_at(const meshgrid_result<ET, ShapeT, Axis> &t,
                        const SubTs &... subs) {
  assert(subscripts_are_valid(t.shape(), subs...));
  return (ET)std::get<Axis>(std::forward_as_tuple(subs...));
}

// meshgrid
template <class ET, class ST, class... SizeTs, class K, K Axis>
constexpr auto meshgrid(const tensor_shape<ST, SizeTs...> &s,
                        const const_ints<K, Axis> &) {
  return meshgrid_result<ET, tensor_shape<ST, SizeTs...>, Axis>(s);
}

// meshgrid
namespace detail {
template <class ET, class ShapeT, size_t... Is>
constexpr auto _meshgrid(const ShapeT &s, const const_ints<size_t, Is...> &) {
  return std::tuple<meshgrid_result<ET, ShapeT, Is>...>(
      meshgrid_result<ET, ShapeT, Is>(s)...);
}
}
template <class ET, class ST, class... SizeTs>
constexpr auto meshgrid(const tensor_shape<ST, SizeTs...> &s) {
  return detail::_meshgrid<ET>(s, make_const_sequence_for<SizeTs...>());
}

/// coordinate
// coordinate_result
template <class ET, class ShapeT>
class coordinate_result
    : public tensor_base<
          tensor<ET, tensor_shape<size_t, const_size<ShapeT::rank>>>, ShapeT,
          coordinate_result<ET, ShapeT>> {
public:
  constexpr explicit coordinate_result(const ShapeT &s) : _shape(s) {}
  constexpr const ShapeT &shape() const { return _shape; }

private:
  ShapeT _shape;
};

// shape_of
template <class ET, class ShapeT>
constexpr decltype(auto) shape_of(const coordinate_result<ET, ShapeT> &t) {
  return t.shape();
}

// element_at
template <class ET, class ShapeT, class... SubTs>
constexpr auto element_at(const coordinate_result<ET, ShapeT> &t,
                          const SubTs &... subs) {
  assert(subscripts_are_valid(t.shape(), subs...));
  return tensor<ET, tensor_shape<size_t, const_size<ShapeT::rank>>>(
      (ET)subs...);
}

// coordinate
template <class ET, class ST, class... SizeTs>
constexpr auto coordinate(const tensor_shape<ST, SizeTs...> &s) {
  return coordinate_result<ET, tensor_shape<ST, SizeTs...>>(s);
}

/// cart_prod
// cart_prod_result
template <class TupleT, class ShapeT, class... Ts>
class cart_prod_result
    : public tensor_base<TupleT, ShapeT,
                         cart_prod_result<TupleT, ShapeT, Ts...>> {
public:
  constexpr cart_prod_result(Ts &&... ins) : inputs(std::forward<Ts>(ins)...) {}

public:
  std::tuple<Ts...> inputs;
};

// shape_of
namespace detail {
template <class CartProdT, size_t... Is>
constexpr decltype(auto)
_shape_of_cart_prod_result_seq(CartProdT &cp,
                               const const_ints<size_t, Is...> &) {
  return make_shape(std::get<Is>(cp.inputs).numel()...);
}
}
template <class TupleT, class ShapeT, class... Ts>
constexpr decltype(auto)
shape_of(const cart_prod_result<TupleT, ShapeT, Ts...> &t) {
  return detail::_shape_of_cart_prod_result_seq(
      t, make_const_sequence_for<Ts...>());
}

// element_at
namespace detail {
template <class CartProdT, class SubsTupleT, size_t... Is>
constexpr auto
_element_at_cart_prod_result_seq(CartProdT &&t, SubsTupleT &&subs,
                                 const const_ints<size_t, Is...> &) {
  return as_tuple(
      element_at_index(std::get<Is>(t.inputs), std::get<Is>(subs))...);
}
}
template <class TupleT, class ShapeT, class... Ts, class... SubTs>
constexpr auto element_at(const cart_prod_result<TupleT, ShapeT, Ts...> &t,
                          const SubTs &... subs) {
  assert(subscripts_are_valid(t.shape(), subs...));
  return detail::_element_at_cart_prod_result_seq(
      t, std::forward_as_tuple(subs...), make_const_sequence_for<Ts...>());
}

// cart_prod
namespace detail {
template <class T> struct _get_value_type_helper {
  using _t = decltype(std::declval<T>().get_value_type());
  using type = std::decay_t<typename _t::type>;
};

template <class... Ts, class... TTs> constexpr auto _cart_prod(TTs &&... tts) {
  using shape_t = decltype(make_shape(tts.numel()...));
  return cart_prod_result<
      std::tuple<typename _get_value_type_helper<TTs>::type...>, shape_t,
      TTs...>(std::forward<TTs>(tts)...);
}
}
}
