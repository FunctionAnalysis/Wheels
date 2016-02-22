#pragma once

#include "base.hpp"

namespace wheels {

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

// meshgrid_at
template <class ET = size_t, class ST, class... SizeTs, class K, K Axis>
constexpr auto meshgrid_at(const tensor_shape<ST, SizeTs...> &s,
                           const const_ints<K, Axis> &) {
  return meshgrid_result<ET, tensor_shape<ST, SizeTs...>, Axis>(s);
}

// meshgrid
namespace details {
template <class ET, class ShapeT, size_t... Is>
constexpr auto _meshgrid(const ShapeT &s, const const_ints<size_t, Is...> &) {
  return std::forward_as_tuple(meshgrid_result<ET, ShapeT, Is>(s)...);
}
}
template <class ET = size_t, class ST, class... SizeTs>
constexpr auto meshgrid(const tensor_shape<ST, SizeTs...> &s) {
  return details::_meshgrid<ET>(s, make_const_sequence_for<SizeTs...>());
}
}
