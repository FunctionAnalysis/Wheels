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
template <class InputShapeT, class InputET, class InputT, class InputTT,
          class ExtShapeT, class ExtFunT, size_t... ExtIs>
constexpr auto
_extend_tensor_by(const tensor_base<InputET, InputShapeT, InputT> &,
                  InputTT &&input, const ExtShapeT &extshape, ExtFunT extfun,
                  const const_ints<size_t, ExtIs...> &) {
  using ele_t = std::decay_t<decltype(
      extfun(std::declval<InputET>(),
             always<size_t, 0, const_index<ExtIs>>::value...))>;
  using shape_t = std::decay_t<decltype(
      cat2(std::declval<InputShapeT>(), std::declval<ExtShapeT>()))>;
  return extend_result<ele_t, shape_t, InputTT, ExtShapeT, ExtFunT>(
      forward<InputTT>(input), extshape, extfun);
}
}

// extend_by
template <class InputT, class ST, class... SizeTs, class ExtFunT>
constexpr auto extend_by(InputT &&input, const tensor_shape<ST, SizeTs...> &es,
                         ExtFunT ef)
    -> decltype(details::_extend_tensor_by(input, forward<InputT>(input), es,
                                           ef, make_rank_sequence(es))) {
  return details::_extend_tensor_by(input, forward<InputT>(input), es, ef,
                                    make_rank_sequence(es));
}

// extend_as_repeated
template <class InputT, class ST, class... SizeTs>
constexpr auto extend_as_repeated(InputT &&input,
                                  const tensor_shape<ST, SizeTs...> &es)
    -> decltype(extend_by(forward<InputT>(input), es,
                          extend_result_ops::as_repeated())) {
  return extend_by(forward<InputT>(input), es,
                   extend_result_ops::as_repeated());
}

// extend_as_subtensor
template <class InputT> inline auto extend_as_subtensor(InputT &&input) {
  assert(input.numel() > 0);
  const auto extshape = element_at_index(input, 0).shape();
  return extend_by(forward<InputT>(input), extshape,
                   extend_result_ops::as_subtensor());
}
}
