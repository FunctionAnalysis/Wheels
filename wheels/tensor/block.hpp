#pragma once

#include "base.hpp"

namespace wheels {

// block_view
template <class ET, class ShapeT, class InputT, class... RangeTs>
class block_view
    : public tensor_base<ET, ShapeT,
                         block_view<ET, ShapeT, InputT, RangeTs...>> {
  static_assert(ShapeT::rank == sizeof...(RangeTs),
                "number of ranges mismatch with shape rank");

public:
  constexpr explicit block_view(InputT &&in, const RangeTs &... rs)
      : input(forward<InputT>(in)), ranges(rs...) {}

  // operator=
  template <class AnotherT>
  block_view &operator=(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
    return *this;
  }

public:
  InputT input;
  std::tuple<RangeTs...> ranges;
};

// shape_of
namespace details {
template <class BlockViewT, size_t... Is>
constexpr auto _shape_of_block_view(BlockViewT &bv,
                                    const const_ints<size_t, Is...> &) {
  return make_shape(std::get<Is>(bv.ranges).size()...);
}
}
template <class ET, class ShapeT, class InputT, class... RangeTs>
constexpr auto shape_of(const block_view<ET, ShapeT, InputT, RangeTs...> &t) {
  return details::_shape_of_block_view(t,
                                       make_const_sequence_for<RangeTs...>());
}

// element_at
namespace details {
template <class BlockViewT, class SubsTupleT, size_t... Is>
constexpr decltype(auto)
_element_at_block_view(BlockViewT &bv, SubsTupleT &&subs,
                       const const_ints<size_t, Is...> &) {
  return element_at(bv.input,
                    std::get<Is>(bv.ranges).begin() + std::get<Is>(subs)...);
}
}
template <class ET, class ShapeT, class InputT, class... RangeTs,
          class... SubTs>
constexpr decltype(auto)
element_at(const block_view<ET, ShapeT, InputT, RangeTs...> &t,
           const SubTs &... subs) {
  return details::_element_at_block_view(t, std::forward_as_tuple(subs...),
                                         make_const_sequence_for<RangeTs...>());
}
template <class ET, class ShapeT, class InputT, class... RangeTs,
          class... SubTs>
decltype(auto) element_at(block_view<ET, ShapeT, InputT, RangeTs...> &t,
                          const SubTs &... subs) {
  return details::_element_at_block_view(t, std::forward_as_tuple(subs...),
                                         make_const_sequence_for<RangeTs...>());
}

// block_at
namespace details {
template <class ET, class ShapeT, class T, class TT, size_t... Is,
          class... RangeTs>
constexpr auto _block_at_direct(const tensor_base<ET, ShapeT, T> &, TT &&t,
                                const const_ints<size_t, Is...> &,
                                const RangeTs &... ranges) {
  using shape_t = decltype(make_shape(ranges.size()...));
  return block_view<ET, shape_t, TT, RangeTs...>(forward<TT>(t), ranges...);
}

template <class ET, class ShapeT, class T, class... OldRangeTs, class TT,
          size_t... Is, class... RangeTs>
constexpr auto
_block_at_direct(const block_view<ET, ShapeT, T, OldRangeTs...> &, TT &&t,
                 const const_ints<size_t, Is...> &seq,
                 const RangeTs &... ranges) {
  return _block_at_direct(
      t.input, forward<TT>(t).input, seq,
      make_range(std::get<Is>(t.ranges).begin() + ranges.begin(),
                 std::get<Is>(t.ranges).begin() + ranges.end())...);
}

// parse the length symbol
template <class TT, size_t... Is, class... RangeTs>
constexpr auto _block_at(TT &&t, const const_ints<size_t, Is...> &seq,
                         const RangeTs &... ranges) {
  return _block_at_direct(
      t, forward<TT>(t), seq,
      make_range(
          details::_eval_index_expr(ranges.begin(), t.size(const_index<Is>())),
          details::_eval_index_expr(ranges.end(),
                                    t.size(const_index<Is>())))...);
}

// transform nonrange indices to ranges
template <class RangeOrIndexT>
constexpr std::enable_if_t<is_range<RangeOrIndexT>::value,
                           const RangeOrIndexT &>
_unify_to_range(const RangeOrIndexT &roi) {
  return roi;
}
template <class RangeOrIndexT,
          class = std::enable_if_t<!is_range<RangeOrIndexT>::value>>
constexpr auto _unify_to_range(const RangeOrIndexT &roi) {
  return span(roi, const_index<1>());
}
}

template <class T, class... RangeOrIndexTs>
constexpr auto block_at(T &&t, const RangeOrIndexTs &... rois)
    -> decltype(details::_block_at(forward<T>(t),
                                   make_const_sequence_for<RangeOrIndexTs...>(),
                                   details::_unify_to_range(rois)...)) {
  return details::_block_at(forward<T>(t),
                            make_const_sequence_for<RangeOrIndexTs...>(),
                            details::_unify_to_range(rois)...);
}
}