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
          ShapeT, ET, void, remap_result<ShapeT, ET, T, MapFunT, IPMethod>> {
public:
  constexpr remap_result(T &&in, const ShapeT &s, MapFunT m, ET o)
      : _input(forward<T>(in)), _output_shape(s), _subs_output2input(m),
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
constexpr ET _element_at_remap_result_seq(
    const remap_result<ShapeT, ET, T, MapFunT, round_interpolate> &r,
    const_ints<size_t, Is...>, const SubTs &... subs) {
  return r.input().at_or(
      r.outlier_value(),
      static_cast<size_t>(std::round(r.float_subs_in_input(subs...)[Is]))...);
}


template <class T1, class T2>
constexpr auto _linear_interpolate(T1 &&p1, T2 &&p2, double c) {
  return forward<T1>(p1) * (1.0 - c) + forward<T2>(p2) * c;
}

// _linear_interpolate
template <class SamplerFunT, class DistToZeroFunT, class... SubTs>
constexpr auto _linear_interpolated_sampling(const_size<0>,
                                             SamplerFunT &samplerfun,
                                             DistToZeroFunT &dist0fun,
                                             const SubTs &... subs) {
  return samplerfun(subs...);
}
template <size_t Undetermined, class SamplerFunT, class DistToZeroFunT,
          class... SubTs>
constexpr auto
_linear_interpolated_sampling(const_size<Undetermined>, SamplerFunT &samplerfun,
                              DistToZeroFunT &dist0fun, const SubTs &... subs) {
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
_element_at_ceil_or_floor_helper(T &t, E &&otherwise, SubsInputT &subs,
                                 NoYesTupleT &noyes,
                                 const_ints<size_t, Is...>) {
  return t.at_or(
      forward<E>(otherwise),
      static_cast<size_t>(conditional(std::get<Is>(noyes), std::ceil(subs[Is]),
                                      std::floor(subs[Is])))...);
}

// _element_at_remap_result_seq using linear_interpolate
template <class ShapeT, class ET, class T, class MapFunT, size_t... Is,
          class... SubTs>
ET _element_at_remap_result_seq(
    const remap_result<ShapeT, ET, T, MapFunT, linear_interpolate> &r,
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
  return remap_result<tensor_shape<ToST, ToSizeTs...>, ET, const T &, MapFunT,
                      IPMethod>(t.derived(), toshape, mapfun,
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
  return remap_result<tensor_shape<ToST, ToSizeTs...>, ET, T, MapFunT,
                      IPMethod>(move(t.derived()), toshape, mapfun,
                                forward<ET2>(outlier));
}
}
