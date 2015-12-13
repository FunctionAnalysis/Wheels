#pragma once

#include "base.hpp"
#include "ewise_ops.hpp"
#include "tensor.hpp"

namespace wheels {

// interpolate_method
enum interpolate_method_enum { round_interpolate, linear_interpolate };
template <interpolate_method_enum IM>
using interpolate_method = const_ints<interpolate_method_enum, IM>;

// remap_result
template <class ShapeT, class ET, class T, class MapFunT,
          interpolate_method_enum IPMethod>
class remap_result
    : public tensor_op_result_base<
          ShapeT, ET, void, remap_result<ShapeT, ET, T, MapT, IPMethod>> {
public:
  constexpr remap_result(T &&in, const ShapeT &s, MapFunT m, ET o)
      : _input(forward<T>(in)), _this_shape(s), _subs_this2input(m),
        _outlier_val(o) {}

  constexpr const ShapeT &shape() const { return _this_shape; }
  constexpr const T &input() const { return _input; }
  template <class... SubTs>
  constexpr decltype(auto) float_subs_in_input(const SubTs &... subs) const {
    return _subs_this2input(subs...);
  }
  constexpr const ET &outlier_value() const { return _outlier_val; }

private:
  T _input;
  ShapeT _this_shape;
  MapFunT _subs_this2input;
  ET _outlier_val;
};

// shape_of
template <class ShapeT, class ET, class T, class MapFunT,
          interpolate_method_enum IPMethod>
constexpr const ShapeT &
shape_of(const remap_result<ShapeT, ET, T, MapFunT, IPMethod> &r) {
  return r.shape();
}

namespace details {
// _element_at_remap_result_seq using round_interpolate
template <class ShapeT, class ET, class T, class MapFunT, size_t... Is,
          class... SubTs>
ET _element_at_remap_result_seq(
    const remap_result<ShapeT, ET, T, MapFunT, round_interpolate> &r,
    const_ints<size_t, Is...>, const SubTs &subs...) {
  return r.input().at_or(
      r.outlier_value(),
      static_cast<size_t>(std::round(r.float_subs_in_input(subs...)[Is]))...);
}

// _linear_interpolate
template <class ET, class SamplerFunT, class DistToZeroFunT, class... SubTs>
ET _linear_interpolate(const_size<0>, SamplerFunT &fun,
                       DistToZeroFunT &dist0fun, const SubTs &... subs) {
  return fun(subs...);
}
template <class ET, size_t Undetermined, class SamplerFunT,
          class DistToZeroFunT, class... SubTs>
ET _linear_interpolate(const_size<Undetermined>, SamplerFunT &fun,
                       DistToZeroFunT &dist0fun, const SubTs &... subs) {
  double dist0 = dist0fun(const_index<Undetermined - 1>());
  ET v0 = _linear_interpolate(const_size<Undetermined - 1>(), fun, dist0fun, 0,
                              subs...);
  ET v1 = _linear_interpolate(const_size<Undetermined - 1>(), fun, dist0fun, 1,
                              subs...);
  return v0 * (1.0 - dist0) + v1 * dist0;
}

// _element_at_remap_result_seq using linear_interpolate
template <class ShapeT, class ET, class T, class MapFunT, size_t... Is,
          class... SubTs>
ET _element_at_remap_result_seq(
    const remap_result<ShapeT, ET, T, MapFunT, linear_interpolate> &r,
    const_ints<size_t, Is...>, const SubTs &subs...) {
  decltype(auto) subsInput =
      static_ecast<double>(r.float_subs_in_input(subs...));
  return _linear_interpolate<ET>(
      const_size<sizeof...(SubTs)>(),
      [&](const auto &... zero_ones) {
        return r.input().at_or(
            r.outlier_value(),
            static_cast<size_t>(zero_ones == 0 ? std::floor(subsInput[Is])
                                               : std::ceil(subsInput[Is]))...);
      },
      [&](const auto &dim) {
        return subsInput[dim] - std::floor(subsInput[dim]);
      });
}
}

// element_at
template <class ShapeT, class ET, class T, class MapFunT,
          interpolate_method_enum IPMethod, class... SubTs>
constexpr ET element_at(const remap_result<ShapeT, ET, T, MapFunT, IPMethod> &r,
                        const SubTs &... subs) {
  return details::_element_at_remap_result_seq(
      r, make_const_sequence_for<SubTs...>(), subs...);
}

// remap
template <class ToST, class... ToSizeTs, class ShapeT, class ET, class T,
          class MapFunT, class ET2 = ET,
          interpolate_method_enum IPMethod = linear_interpolate>
constexpr auto
remap(const tensor_base<ShapeT, ET, T> &t,
      const tensor_shape<ToST, ToSizeTs...> &toshape, MapFunT mapfun,
      ET2 &&outlier = types<ET2>::zero(),
      interpolate_method<IPMethod> = interpolate_method<IPMethod>()) {
  static_assert(sizeof...(ToSizeTs) == ShapeT::rank, "shape rank mismatch");
  return remap_result<tensor_shape<ToST, ToSizeTs...>, ET, const T &, MapFunT,
                      ET, IPMethod>(t.derived(), toshape, mapfun,
                                    forward<ET2>(outlier));
}

template <class ToST, class... ToSizeTs, class ShapeT, class ET, class T,
          class MapFunT, class ET2 = ET,
          interpolate_method_enum IPMethod = linear_interpolate>
constexpr auto
remap(tensor_base<ShapeT, ET, T> &&t,
      const tensor_shape<ToST, ToSizeTs...> &toshape, MapFunT mapfun,
      ET2 &&outlier = types<ET2>::zero(),
      interpolate_method<IPMethod> = interpolate_method<IPMethod>()) {
  static_assert(sizeof...(ToSizeTs) == ShapeT::rank, "shape rank mismatch");
  return remap_result<tensor_shape<ToST, ToSizeTs...>, ET, T, MapFunT, ET,
                      IPMethod>(move(t.derived()), toshape, mapfun,
                                forward<ET2>(outlier));
}
}
