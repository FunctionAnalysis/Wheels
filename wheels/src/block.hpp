#pragma once

#include "tensor_base.hpp"

#include "block_fwd.hpp"

namespace wheels {

template <class ET, class ShapeT, class InputTensorT,
          class... SubscriptTensorTs>
class block_view
    : public tensor_base<ET, ShapeT, block_view<ET, ShapeT, InputTensorT,
                                                SubscriptTensorTs...>> {
public:
  constexpr block_view(InputTensorT &&in, SubscriptTensorTs &&... subts)
      : input_tensor(std::forward<InputTensorT>(in)),
        subs_tensors(std::forward<SubscriptTensorTs>(subts)...) {}

  // operator=
  template <class AnotherT>
  block_view &operator=(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
    return *this;
  }
  block_view &operator=(const ET &e) {
    fill_elements_with(*this, e);
    return *this;
  }

public:
  InputTensorT input_tensor;
  std::tuple<SubscriptTensorTs...> subs_tensors;
};

// shape_of
namespace details {
template <class SubsViewT, size_t... Is>
constexpr auto _shape_of_block_view(SubsViewT &sv,
                                    const const_ints<size_t, Is...> &) {
  return make_shape(numel_of(std::get<Is>(sv.subs_tensors).derived())...);
}
}
template <class ET, class ShapeT, class InputTensorT,
          class... SubscriptTensorTs>
constexpr auto
shape_of(const block_view<ET, ShapeT, InputTensorT, SubscriptTensorTs...> &t) {
  return details::_shape_of_block_view(
      t, make_const_sequence_for<SubscriptTensorTs...>());
}

// element_at
namespace details {
template <class SubsViewT, class SubsTupleT, size_t... Is>
constexpr decltype(auto)
_element_at_subscript_view_seq(SubsViewT &sv, SubsTupleT &&subs,
                               const const_ints<size_t, Is...> &) {
  return element_at(
      sv.input_tensor,
      element_at_index(std::get<Is>(sv.subs_tensors), std::get<Is>(subs))...);
}
}
template <class ET, class ShapeT, class InputTensorT,
          class... SubscriptTensorTs, class... SubTs>
constexpr decltype(auto)
element_at(const block_view<ET, ShapeT, InputTensorT, SubscriptTensorTs...> &t,
           const SubTs &... subs) {
  return details::_element_at_subscript_view_seq(
      t, std::forward_as_tuple(subs...),
      make_const_sequence_for<SubscriptTensorTs...>());
}

// at_block
namespace details {
template <class InET, class InShapeT, class InT, class InTT,
          class... SubsTensorTs>
constexpr auto _at_block(const tensor_base<InET, InShapeT, InT> &, InTT &&in,
                         SubsTensorTs &&... sts) {
  using shape_t = std::decay_t<decltype(make_shape(sts.numel()...))>;
  return block_view<InET, shape_t, InTT, SubsTensorTs...>(
      std::forward<InTT>(in), std::forward<SubsTensorTs>(sts)...);
}
}
}
