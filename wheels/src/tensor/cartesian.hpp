#pragma once

#include "base.hpp"
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
  return (ET)std::get<Axis>(std::forward_as_tuple(subs...));
}

// meshgrid
template <class ET = size_t, class ST, class... SizeTs, class K, K Axis>
constexpr auto meshgrid(const tensor_shape<ST, SizeTs...> &s,
                        const const_ints<K, Axis> &) {
  return meshgrid_result<ET, tensor_shape<ST, SizeTs...>, Axis>(s);
}

// meshgrid
namespace details {
template <class ET, class ShapeT, size_t... Is>
constexpr auto _meshgrid(const ShapeT &s, const const_ints<size_t, Is...> &) {
  return safe_forward_as_tuple(meshgrid_result<ET, ShapeT, Is>(s)...);
}
}
template <class ET = size_t, class ST, class... SizeTs>
constexpr auto meshgrid(const tensor_shape<ST, SizeTs...> &s) {
  return details::_meshgrid<ET>(s, make_const_sequence_for<SizeTs...>());
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
  return tensor<ET, tensor_shape<size_t, const_size<ShapeT::rank>>>(
      (ET)subs...);
}

// coordinate
template <class ET = size_t, class ST, class... SizeTs>
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
namespace details {
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
  return details::_shape_of_cart_prod_result_seq(
      t, make_const_sequence_for<Ts...>());
}

// element_at
namespace details {
template <class CartProdT, class SubsTupleT, size_t... Is>
constexpr auto
_element_at_cart_prod_result_seq(CartProdT &t, SubsTupleT &subs,
                                 const const_ints<size_t, Is...> &) {
  return safe_forward_as_tuple(
      element_at_index(std::get<Is>(t.inputs), std::get<Is>(subs))...);
}
}
template <class TupleT, class ShapeT, class... Ts, class... SubTs>
constexpr auto element_at(const cart_prod_result<TupleT, ShapeT, Ts...> &t,
                          const SubTs &... subs) {
  return details::_element_at_cart_prod_result_seq(
      t, std::forward_as_tuple(subs...), make_const_sequence_for<Ts...>());
}

// cart_prod
namespace details {
template <class... Ts, class... TTs> constexpr auto _cart_prod(TTs &&... tts) {
  using shape_t = decltype(make_shape(tts.numel()...));
  return cart_prod_result<std::tuple<type_t(tts.get_value_type())...>, shape_t,
                          TTs...>(std::forward<TTs>(tts)...);
}
}
template <class... TTs>
constexpr auto cart_prod(TTs &&... ts)
    -> decltype(details::_cart_prod(std::forward<TTs>(ts)...)) {
  return details::_cart_prod(std::forward<TTs>(ts)...);
}
}
