#pragma once

#include "tensor_base_fwd.hpp"

namespace wheels {

// subtensor_view
template <class ET, class SubShapeT, class InputT, size_t FixedRank>
class subtensor_view;

namespace details {
template <class ET, class ShapeT, class T, class TT, class... SubTs>
constexpr auto _subtensor_at(const tensor_base<ET, ShapeT, T> &, TT &&input,
                             const SubTs &... subs);
}

template <class T, class... SubTs>
constexpr auto subtensor_at(T &&input, const SubTs &... subs)
    -> decltype(details::_subtensor_at(input, std::forward<T>(input),
                                       subs...)) {
  return details::_subtensor_at(input, std::forward<T>(input), subs...);
}

// downgrade_view
template <class ET, class ShapeT, class InputT, size_t FixedRank>
class downgrade_view;

template <class ET, class ShapeT, class InputT, class ExtShapeT, class ExtFunT>
class upgrade_result;
namespace upgrade_result_ops {
struct as_subtensor;
}

namespace details {
template <class ET, class ShapeT, class InputT, class TT, size_t FixedRank>
constexpr auto _downgrade(const tensor_base<ET, ShapeT, InputT> &, TT &&input,
                          const const_size<FixedRank> &);
// downgrade an upgraded tensor
template <class ET, class ShapeT, class InputT, class ExtShapeT, class TT>
constexpr decltype(auto)
_downgrade(const upgrade_result<ET, ShapeT, InputT, ExtShapeT,
                                upgrade_result_ops::as_subtensor> &,
           TT &&input, const const_size<ExtShapeT::rank> &);
}
template <class InputT, class K, K FixedRank>
constexpr auto downgrade(InputT &&input, const const_ints<K, FixedRank> &r)
    -> decltype(details::_downgrade(input, std::forward<InputT>(input),
                                    const_size<(size_t)FixedRank>())) {
  return details::_downgrade(input, std::forward<InputT>(input),
                             const_size<(size_t)FixedRank>());
}
}
