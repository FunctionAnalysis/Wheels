#pragma once

#include "tensor_base.hpp"
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
  auto sampler_fun = [&r, &subsInput](auto &&... noyeses) {
    return _element_at_ceil_or_floor_helper(
        r.input(), r.outlier_value(), subsInput,
        std::forward_as_tuple(noyeses...),
        make_const_sequence(const_size<sizeof...(noyeses)>()));
  };
  auto dist0_fun = [&subsInput](const auto &dim) {
    return subsInput[dim] - std::floor(subsInput[dim]);
  };
  return ET(_linear_interpolated_sampling(
      const_size<input_rank>(), std::move(sampler_fun), std::move(dist0_fun)));
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
namespace details {
template <class ToST, class... ToSizeTs, class ET, class ShapeT, class T,
          class TT, class MapFunT, class ET2, interpolate_method_enum IPMethod>
constexpr auto _remap(const tensor_base<ET, ShapeT, T> &, TT &&t,
                      const tensor_shape<ToST, ToSizeTs...> &toshape,
                      MapFunT mapfun, ET2 &&outlier,
                      const interpolate_method<IPMethod> &) {
  return remap_result<ET, tensor_shape<ToST, ToSizeTs...>, TT, MapFunT,
                      IPMethod>(std::forward<TT>(t), toshape, mapfun,
                                std::forward<ET2>(outlier));
}
template <class ToST, class... ToSizeTs, class ET, class ShapeT, class T,
          class TT, class MapFunT, interpolate_method_enum IPMethod>
constexpr auto _remap(const tensor_base<ET, ShapeT, T> &, TT &&t,
                      const tensor_shape<ToST, ToSizeTs...> &toshape,
                      MapFunT mapfun, const interpolate_method<IPMethod> &) {
  return remap_result<ET, tensor_shape<ToST, ToSizeTs...>, TT, MapFunT,
                      IPMethod>(std::forward<TT>(t), toshape, mapfun,
                                types<ET>::zero());
}
}

namespace details {
template <class FromShapeT, class ToShapeT> struct _resample_map_functor {
  static_assert(FromShapeT::rank == ToShapeT::rank, "shape ranks mismatch!");
  FromShapeT from_shape;
  ToShapeT to_shape;
  template <class NewSubsTupleT, size_t... Is>
  constexpr auto _invoke_seq(NewSubsTupleT &&new_subs,
                             const const_ints<size_t, Is...> &) const {
    return std::array<double, sizeof...(Is)>{
        {static_cast<double>(std::get<Is>(new_subs)) *
         (from_shape.at(const_index<Is>()) - 1.0) /
         (to_shape.at(const_index<Is>()) - 1.0)...}};
  }
  template <class... SubTs> constexpr auto operator()(SubTs... new_subs) const {
    static_assert(FromShapeT::rank == sizeof...(new_subs),
                  "subscripts num and shape rank mismatch!");
    return _invoke_seq(std::forward_as_tuple(new_subs...),
                       make_const_sequence_for<SubTs...>());
  }
};
template <class FromShapeT, class ToShapeT>
_resample_map_functor<FromShapeT, ToShapeT>
_make_resample_map_functor(const FromShapeT &from_shape,
                           const ToShapeT &to_shape) {
  return _resample_map_functor<FromShapeT, ToShapeT>{from_shape, to_shape};
}
}

}