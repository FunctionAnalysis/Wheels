#pragma once

#include "tensor_base_fwd.hpp"

namespace wheels {

template <class ET, class ShapeT, class InputT, class SubsMapFunT>
class reformulate_result;

namespace details {
template <class ET, class ShapeT, class T, class TT, class NewShapeT,
          class SubsMapFunT>
constexpr auto _reformulate(const tensor_base<ET, ShapeT, T> &, TT &&t,
                            const NewShapeT &s, SubsMapFunT fun);
}
// reformulate
template <class T, class ST, class... SizeTs, class SubsMapFunT>
constexpr auto reformulate(T &&t, const tensor_shape<ST, SizeTs...> &shape,
                           SubsMapFunT fun)
    -> decltype(details::_reformulate(t, std::forward<T>(t), shape, fun)) {
  return details::_reformulate(t, std::forward<T>(t), shape, fun);
}

namespace details {
template <class ET, class ShapeT, class T, class TT, class RepsTupleT,
          size_t... Is>
constexpr auto _repeat_impl(const tensor_base<ET, ShapeT, T> &, TT &&t,
                            RepsTupleT &&rps,
                            const const_ints<size_t, Is...> &);
}

// repeat
template <class T, class... RepTs>
constexpr auto repeat(T &&t, const RepTs &... reps)
    -> decltype(details::_repeat_impl(t, std::forward<T>(t),
                                      std::forward_as_tuple(reps...),
                                      make_const_sequence_for<RepTs...>())) {
  return details::_repeat_impl(t, std::forward<T>(t),
                               std::forward_as_tuple(reps...),
                               make_const_sequence_for<RepTs...>());
}
}