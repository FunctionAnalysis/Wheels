#pragma once

#include "base.hpp"

namespace wheels {

// reformulate_result
template <class ET, class ShapeT, class InputT, class SubsMapFunT>
class reformulate_result
    : public tensor_base<ET, ShapeT,
                         reformulate_result<ET, ShapeT, InputT, SubsMapFunT>> {
public:
  constexpr explicit reformulate_result(const ShapeT &s, SubsMapFunT f,
                                        InputT &&in)
      : _shape(s), _subs_map_fun(f), _input(forward<InputT>(in)) {}

  constexpr const ShapeT &shape() const { return _shape; }
  constexpr const InputT &input() const & { return _input; }
  InputT &input() & { return _input; }
  InputT &&input() && { return _input; }

  template <class... SubTs>
  constexpr decltype(auto) subs_of_input(const SubTs &... subs) const {
    return _subs_map_fun(subs...);
  }

private:
  ShapeT _shape;
  SubsMapFunT _subs_map_fun;
  InputT _input;
};

// shape_of
template <class ET, class ShapeT, class InputT, class SubsMapFunT>
constexpr const ShapeT &
shape_of(const reformulate_result<ET, ShapeT, InputT, SubsMapFunT> &r) {
  return r.shape();
}

// element_at
namespace details {
template <class InputT, class SubsTupleT, size_t... Is>
constexpr decltype(auto) _element_at_subs_seq(InputT &input, SubsTupleT &&subs,
                                              const_ints<size_t, Is...> &) {
  return element_at(input, std::get<Is>(subs)...);
}
}
template <class ET, class ShapeT, class InputT, class SubsMapFunT,
          class... SubTs>
constexpr decltype(auto)
element_at(const reformulate_result<ET, ShapeT, InputT, SubsMapFunT> &r,
           const SubTs &... subs) {
  return details::_element_at_subs_seq(r.input(), r.subs_of_input(subs...),
                                       make_rank_sequence(r.input().shape()));
}

namespace details {
template <class ET, class ShapeT, class T, class TT, class NewShapeT,
          class SubsMapFunT>
constexpr auto _reformulate(const tensor_base<ET, ShapeT, T> &, TT &&t,
                            const NewShapeT &s, SubsMapFunT fun) {
  return reformulate_result<ET, NewShapeT, TT, SubsMapFunT>(s, fun,
                                                            forward<TT>(t));
}
}
// reformulate
template <class T, class ST, class... SizeTs, class SubsMapFunT>
constexpr auto reformulate(T &&t, const tensor_shape<ST, SizeTs...> &shape,
                           SubsMapFunT fun)
    -> decltype(details::_reformulate(t, forward<T>(t), shape, fun)) {
  return details::_reformulate(t, forward<T>(t), shape, fun);
}

// repeat
namespace details {
template <class ShapeT, class SubsTupleT, size_t... Is>
constexpr auto _repeat_subs_seq(const ShapeT &shape, SubsTupleT &subs,
                                const const_ints<size_t, Is...> &) {
  return std::forward_as_tuple((std::get<Is>(subs) % std::get<Is>(shape))...);
}
template <class ShapeT> struct _repeat_subs_functor {
  ShapeT shape;
  constexpr explicit _repeat_subs_functor(const ShapeT &s) : shape(s) {}
  template <class... SubTs>
  constexpr auto operator()(const SubTs &... subs) const {
    return _repeat_subs_seq(shape, std::forward_as_tuple(subs...),
                            make_const_sequence_for<SubTs...>());
  }
};

template <class ET, class ShapeT, class T, class TT, class RepsTupleT,
          size_t... Is>
constexpr auto _repeat_impl(const tensor_base<ET, ShapeT, T> &, TT &&t,
                            RepsTupleT &&rps,
                            const const_ints<size_t, Is...> &) {
  static_assert(sizeof...(Is) == ShapeT::rank, "wrong number of repeats");
  return _reformulate(
      t, forward<T>(t),
      make_shape((std::get<Is>(rps) * t.size(const_index<Is>()))...),
      _repeat_subs_functor<ShapeT>(t.shape()));
}
}

template <class T, class... RepTs>
constexpr auto repeat(T &&t, const RepTs &... reps)
    -> decltype(details::_repeat_impl(t, forward<T>(t),
                                      std::forward_as_tuple(reps...),
                                      make_const_sequence_for<RepTs...>())) {
  return details::_repeat_impl(t, forward<T>(t), std::forward_as_tuple(reps...),
                               make_const_sequence_for<RepTs...>());
}
}