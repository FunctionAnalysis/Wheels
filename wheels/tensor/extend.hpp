#pragma once

#include "base.hpp"

namespace wheels {

template <class ShapeT, class ET, class InputT, class ExtShapeT, class ExtFunT>
class extend_result
    : public tensor_op_result_base<
          ShapeT, ET, void,
          extend_result<ShapeT, ET, InputT, ExtShapeT, ExtFunT>> {
public:
  extend_result(InputT &&in, const ExtShapeT &es, ExtFunT ef)
      : input(forward<InputT>(in)), ext_shape(es), ext_fun(ef) {}

public:
  InputT input;
  ExtShapeT ext_shape;
  ExtFunT ext_fun;
};

// shape_of
template <class ShapeT, class ET, class InputT, class ExtShapeT, class ExtFunT>
constexpr auto
shape_of(const extend_result<ShapeT, ET, InputT, ExtShapeT, ExtFunT> &r) {
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

template <class ShapeT, class ET, class InputT, class ExtShapeT, class ExtFunT,
          class... SubTs>
constexpr decltype(auto)
element_at(const extend_result<ShapeT, ET, InputT, ExtShapeT, ExtFunT> &r,
           const SubTs &... subs) {
  return details::_element_at_extend_result_seq(
      r, std::forward_as_tuple(subs...), make_rank_sequence(r.input.shape()),
      make_rank_sequence(r.ext_shape));
}

template <class ShapeT, class ET, class InputT, class ExtShapeT, class ExtFunT,
          class... SubTs>
decltype(auto)
element_at(extend_result<ShapeT, ET, InputT, ExtShapeT, ExtFunT> &r,
           const SubTs &... subs) {
  return details::_element_at_extend_result_seq(
      r, std::forward_as_tuple(subs...), make_rank_sequence(r.input.shape()),
      make_rank_sequence(r.ext_shape));
}

// extend_result_op_identity
struct extend_result_op_identity {
  template <class ET, class... SubTs>
  constexpr ET &&operator()(ET &&e, const SubTs &...) const {
    return static_cast<ET &&>(e);
  }
};

// extend_result_op_subtensor
struct extend_result_op_subtensor {
  template <class SubTensorT, class... SubTs>
  constexpr decltype(auto) operator()(SubTensorT &&e,
                                      const SubTs &... subs) const {
    return element_at(forward<SubTensorT>(e), subs...);
  }
};
}
