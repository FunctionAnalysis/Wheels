#pragma once

#include "base.hpp"
#include "tensor.hpp"

#include "remap_fwd.hpp"

namespace wheels {

// remap_result
template <class ET, class ShapeT, class T, class MapFunT,
          interpolate_method_enum IPMethod>
class remap_result
    : public tensor_base<ET, ShapeT,
                         remap_result<ET, ShapeT, T, MapFunT, IPMethod>> {
public:
  constexpr remap_result(T &&in, const ShapeT &s, MapFunT m, ET o)
      : _input(std::forward<T>(in)), _output_shape(s), _subs_output2input(m),
        _outlier_val(o) {}

  constexpr const ShapeT &shape() const { return _output_shape; }
  constexpr const T &input() const { return _input; }
  template <class... SubTs>
  constexpr decltype(auto) float_subs_in_input(const SubTs &... subs) const {
    return _subs_output2input(subs...);
  }
  constexpr const ET &outlier_value() const { return _outlier_val; }

private:
  T _input;
  ShapeT _output_shape;
  MapFunT _subs_output2input;
  ET _outlier_val;
};

// shape_of
template <class ET, class ShapeT, class T, class MapFunT,
          interpolate_method_enum IPMethod>
constexpr const ShapeT &
shape_of(const remap_result<ET, ShapeT, T, MapFunT, IPMethod> &r) {
  return r.shape();
}

namespace details {
// _element_at_remap_result_seq using round_interpolate
template <class ET, class ShapeT, class T, class MapFunT, size_t... Is,
          class... SubTs>
constexpr ET _element_at_remap_result_seq(
    const remap_result<ET, ShapeT, T, MapFunT, round_interpolate> &r,
    const_ints<size_t, Is...>, const SubTs &... subs) {
  return r.input().at_or(
      r.outlier_value(),
      static_cast<size_t>(std::round(r.float_subs_in_input(subs...)[Is]))...);
}

template <class T1, class T2>
constexpr auto _linear_interpolate(T1 &&p1, T2 &&p2, double c) {
  return std::forward<T1>(p1) * (1.0 - c) + std::forward<T2>(p2) * c;
}

// _linear_interpolate
template <class SamplerFunT, class DistToZeroFunT, class... SubTs>
constexpr auto _linear_interpolated_sampling(const_size<0>,
                                             SamplerFunT &&samplerfun,
                                             DistToZeroFunT &&dist0fun,
                                             const SubTs &... subs) {
  return samplerfun(subs...);
}
template <size_t Undetermined, class SamplerFunT, class DistToZeroFunT,
          class... SubTs>
constexpr auto _linear_interpolated_sampling(const_size<Undetermined>,
                                             SamplerFunT &&samplerfun,
                                             DistToZeroFunT &&dist0fun,
                                             const SubTs &... subs) {
  return details::_linear_interpolate(
      _linear_interpolated_sampling(const_size<Undetermined - 1>(), samplerfun,
                                    dist0fun, no(), subs...),
      _linear_interpolated_sampling(const_size<Undetermined - 1>(), samplerfun,
                                    dist0fun, yes(), subs...),
      dist0fun(const_index<Undetermined - 1>()));
}

// _element_at_ceil_or_floor_helper
template <class T, class E, class SubsInputT, class NoYesTupleT, size_t... Is>
constexpr decltype(auto)
_element_at_ceil_or_floor_helper(T &&t, E &&otherwise, SubsInputT &&subs,
                                 NoYesTupleT &&noyes,
                                 const_ints<size_t, Is...>) {
  return t.at_or(
      std::forward<E>(otherwise),
      static_cast<size_t>(conditional(std::get<Is>(noyes), std::ceil(subs[Is]),
                                      std::floor(subs[Is])))...);
}

// _element_at_remap_result_seq using linear_interpolate
template <class ET, class ShapeT, class T, class MapFunT, size_t... Is,
          class... SubTs>
ET _element_at_remap_result_seq(
    const remap_result<ET, ShapeT, T, MapFunT, linear_interpolate> &r,
    const_ints<size_t, Is...>, const SubTs &... subs) {
  static_assert(sizeof...(SubTs) == ShapeT::rank,
                "invalid number of subscripts");
  decltype(auto) subsInput = r.float_subs_in_input(subs...);
  constexpr size_t input_rank =
      std::decay_t<decltype(std::declval<T>().shape())>::rank;
  // the size of subsInput should be same with input_rank
  return (ET)_linear_interpolated_sampling(
      const_size<input_rank>(),
      [&r, &subsInput](auto &&... noyeses) {
        return _element_at_ceil_or_floor_helper(
            r.input(), r.outlier_value(), subsInput,
            std::forward_as_tuple(noyeses...),
            make_const_sequence(const_size<sizeof...(noyeses)>()));
      },
      [&subsInput](const auto &dim) {
        return subsInput[dim] - std::floor(subsInput[dim]);
      });
}
}

// element_at
template <class ET, class ShapeT, class T, class MapFunT,
          interpolate_method_enum IPMethod, class... SubTs>
constexpr ET element_at(const remap_result<ET, ShapeT, T, MapFunT, IPMethod> &r,
                        const SubTs &... subs) {
  return details::_element_at_remap_result_seq(
      r, make_const_sequence_for<SubTs...>(), subs...);
}

// remap
template <class ToST, class... ToSizeTs, class ET, class ShapeT, class T,
          class MapFunT, class ET2, interpolate_method_enum IPMethod>
constexpr auto remap(const tensor_base<ET, ShapeT, T> &t,
                     const tensor_shape<ToST, ToSizeTs...> &toshape,
                     MapFunT mapfun, ET2 &&outlier,
                     const interpolate_method<IPMethod> &) {
  return remap_result<ET, tensor_shape<ToST, ToSizeTs...>, const T &, MapFunT,
                      IPMethod>(t.derived(), toshape, mapfun,
                                std::forward<ET2>(outlier));
}

template <class ToST, class... ToSizeTs, class ET, class ShapeT, class T,
          class MapFunT, class ET2, interpolate_method_enum IPMethod>
constexpr auto remap(tensor_base<ET, ShapeT, T> &&t,
                     const tensor_shape<ToST, ToSizeTs...> &toshape,
                     MapFunT mapfun, ET2 &&outlier,
                     interpolate_method<IPMethod>) {
  return remap_result<ET, tensor_shape<ToST, ToSizeTs...>, T, MapFunT,
                      IPMethod>(std::move(t.derived()), toshape, mapfun,
                                std::forward<ET2>(outlier));
}
}
