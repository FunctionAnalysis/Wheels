#pragma once

#include "base.hpp"

namespace wheels {

template <class ET, class ShapeT, class InputT, class ExtShapeT, class ExtFunT>
class extend_result
    : public tensor_op_result_base<
          ET, ShapeT, void,
          extend_result<ET, ShapeT, InputT, ExtShapeT, ExtFunT>> {
public:
  extend_result(InputT &&in, const ExtShapeT &es, ExtFunT ef)
      : input(forward<InputT>(in)), ext_shape(es), ext_fun(ef) {}

public:
  InputT input;
  ExtShapeT ext_shape;
  ExtFunT ext_fun;
};

// shape_of
template <class ET, class ShapeT, class InputT, class ExtShapeT, class ExtFunT>
constexpr auto
shape_of(const extend_result<ET, ShapeT, InputT, ExtShapeT, ExtFunT> &r) {
  return cat2(r.input.shape(), r.ext_shape);
}

// element_at
namespace details {
template <class ExtendResultT, class SubsTupleT, size_t... OldIs,
          size_t... ExtendedIs>
constexpr decltype(auto)
_element_at_extend_result_seq(ExtendResultT &&r, SubsTupleT &&subs,
                              const const_ints<size_t, OldIs...> &,
                              const const_ints<size_t, ExtendedIs...> &) {
  return r.ext_fun(element_at(r.input, std::get<OldIs>(subs)...),
                   std::get<ExtendedIs + sizeof...(OldIs)>(subs)...);
}
}

template <class ET, class ShapeT, class InputT, class ExtShapeT, class ExtFunT,
          class... SubTs>
constexpr decltype(auto)
element_at(const extend_result<ET, ShapeT, InputT, ExtShapeT, ExtFunT> &r,
           const SubTs &... subs) {
  return details::_element_at_extend_result_seq(
      r, std::forward_as_tuple(subs...), make_rank_sequence(r.input.shape()),
      make_rank_sequence(r.ext_shape));
}

template <class ET, class ShapeT, class InputT, class ExtShapeT, class ExtFunT,
          class... SubTs>
decltype(auto)
element_at(extend_result<ET, ShapeT, InputT, ExtShapeT, ExtFunT> &r,
           const SubTs &... subs) {
  return details::_element_at_extend_result_seq(
      r, std::forward_as_tuple(subs...), make_rank_sequence(r.input.shape()),
      make_rank_sequence(r.ext_shape));
}

namespace extend_result_ops {
// as_repeated
struct as_repeated {
  template <class ET, class... SubTs>
  constexpr ET &&operator()(ET &&e, const SubTs &...) const {
    return static_cast<ET &&>(e);
  }
};

// as_subtensor
struct as_subtensor {
  template <class SubTensorT, class... SubTs>
  constexpr decltype(auto) operator()(SubTensorT &&e,
                                      const SubTs &... subs) const {
    return element_at(forward<SubTensorT>(e), subs...);
  }
};
}

namespace details {
template <class ET, class InputShapeT, class InputET, class InputT,
          class InputTT, class ExtShapeT, class ExtFunT, size_t... ExtIs>
constexpr auto
_extend_tensor_by(const tensor_base<InputET, InputShapeT, InputT> &,
                  InputTT &&input, const ExtShapeT &extshape, ExtFunT extfun,
                  const const_ints<size_t, ExtIs...> &) {
  using ele_t = ET;
  using shape_t = std::decay_t<decltype(
      cat2(std::declval<InputShapeT>(), std::declval<ExtShapeT>()))>;
  return extend_result<ele_t, shape_t, InputTT, ExtShapeT, ExtFunT>(
      forward<InputTT>(input), extshape, extfun);
}
}

// extend_by
template <class ET, class InputT, class ST, class... SizeTs, class ExtFunT>
constexpr auto extend_by(InputT &&input, const tensor_shape<ST, SizeTs...> &es,
                         ExtFunT ef)
    -> decltype(details::_extend_tensor_by<ET>(input, forward<InputT>(input),
                                               es, ef,
                                               make_rank_sequence(es))) {
  return details::_extend_tensor_by<ET>(input, forward<InputT>(input), es, ef,
                                        make_rank_sequence(es));
}

// extend_as_repeated
namespace details {
template <class ET, class ShapeT, class T, class InputT, class ST,
          class... SizeTs>
constexpr auto _extend_as_repeated(const tensor_base<ET, ShapeT, T> &,
                                   InputT &&input,
                                   const tensor_shape<ST, SizeTs...> &es) {
  return extend_by<ET>(forward<InputT>(input), es,
                       extend_result_ops::as_repeated());
}
}
template <class InputT, class ST, class... SizeTs>
constexpr auto extend_as_repeated(InputT &&input,
                                  const tensor_shape<ST, SizeTs...> &es) {
  return details::_extend_as_repeated(input, forward<InputT>(input), es);
}

template <class ET, class ShapeT, class T, size_t FixedRank>
class subtensor_view;
template <class ET, class ShapeT, class T, size_t FixedRank>
class tensor_subwise_view;

// extend_as_subtensor
namespace details {
template <class ET, class ShapeT, class T, class ET2, class ShapeT2, class T2,
          class InputT>
constexpr decltype(auto)
_extend_as_subtensor(const tensor_base<ET, ShapeT, T> &et,
                     const tensor_base<ET2, ShapeT2, T2> &, InputT &&input) {
  return extend_by<ET>(forward<InputT>(input), et.shape(),
                       extend_result_ops::as_subtensor());
}
// extend a subwised tensor
template <class ET, class ShapeT, class T, class ShapeT2, class T2,
          size_t FixedRank, class InputT>
constexpr decltype(auto)
_extend_as_subtensor(const subtensor_view<ET, ShapeT, T, FixedRank> &et,
                     const tensor_subwise_view<ET, ShapeT2, T2, FixedRank> &,
                     InputT &&input) {
  return forward<InputT>(input).input;
}
}
template <class InputT>
inline decltype(auto) extend_as_subtensor(InputT &&input) {
  assert(input.numel() > 0);
  return details::_extend_as_subtensor(element_at_index(input, 0), input,
                                       forward<InputT>(input));
}
}
