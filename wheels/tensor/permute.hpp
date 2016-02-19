#pragma once

#include "tensor.hpp"

namespace wheels {

// permute_result
template <class ET, class ShapeT, class T, size_t... Inds>
class permute_result
    : public tensor_op_result_base<ET, ShapeT, void,
                                   permute_result<ET, ShapeT, T, Inds...>> {
  static_assert(sizeof...(Inds) == ShapeT::rank,
                "invalid number of inds in permute");

public:
  using value_type = ET;
  using shape_type = ShapeT;
  constexpr explicit permute_result(T &&in) : input(forward<T>(in)) {}

public:
  T input;
};

// shape_of
template <class ET, class ShapeT, class T, size_t... Inds>
constexpr auto shape_of(const permute_result<ET, ShapeT, T, Inds...> &m) {
  return ::wheels::permute(m.input.shape(), const_index<Inds>()...);
}

// element_at
namespace details {
template <class ET, class ShapeT, class T, size_t... Inds, class SubsTupleT,
          size_t... Is>
constexpr decltype(auto)
_element_at_permute_result_seq(const permute_result<ET, ShapeT, T, Inds...> &m,
                               SubsTupleT &&subs, const_ints<size_t, Is...>) {
  return ::wheels::element_at(
      m.input,
      std::get<decltype(::wheels::find_first_of(
          const_ints<size_t, Inds...>(), const_index<Is>()))::value>(subs)...);
}
}
template <class ET, class ShapeT, class T, size_t... Inds, class... SubTs>
constexpr decltype(auto)
element_at(const permute_result<ET, ShapeT, T, Inds...> &m,
           const SubTs &... subs) {
  return details::_element_at_permute_result_seq(
      m, std::forward_as_tuple(subs...), make_const_sequence_for<SubTs...>());
}

// unordered
template <class FunT, class ET, class ShapeT, class T, size_t... Inds>
bool for_each_element(behavior_flag<unordered> o, FunT &&fun,
                      const permute_result<ET, ShapeT, T, Inds...> &m) {
  return for_each_element(o, forward<FunT>(fun), m.input);
}

// break_on_false
template <class FunT, class ET, class ShapeT, class T, size_t... Inds>
bool for_each_element(behavior_flag<break_on_false> o, FunT &&fun,
                      const permute_result<ET, ShapeT, T, Inds...> &m) {
  return for_each_element(o, forward<FunT>(fun), m.input);
}

// nonzero_only
template <class FunT, class ET, class ShapeT, class T, size_t... Inds>
bool for_each_element(behavior_flag<nonzero_only> o, FunT &&fun,
                      const permute_result<ET, ShapeT, T, Inds...> &m) {
  return for_each_element(o, forward<FunT>(fun), m.input);
}

// reduce_elements
template <class ET, class ShapeT, class T, class E, class ReduceT,
          size_t... Inds>
constexpr E reduce_elements(const permute_result<ET, ShapeT, T, Inds...> &t,
                            E initial, ReduceT &&red) {
  return reduce_elements(t.input, move(initial), forward<ReduceT>(red));
}

// norm_squared
template <class ET, class ShapeT, class T, size_t... Inds>
constexpr auto norm_squared(const permute_result<ET, ShapeT, T, Inds...> &t) {
  return norm_squared(t.input);
}

// bool all(s)
template <class ET, class ShapeT, class T, size_t... Inds>
constexpr bool all_of(const permute_result<ET, ShapeT, T, Inds...> &t) {
  return all_of(t.input);
}
// bool any(s)
template <class ET, class ShapeT, class T, size_t... Inds>
constexpr bool any_of(const permute_result<ET, ShapeT, T, Inds...> &t) {
  return any_of(t.input);
}

namespace details {
template <class ET, class ShapeT, class T, size_t... Inds>
constexpr permute_result<ET, ShapeT, T, Inds...> &&
_simplify_permute_impl(permute_result<ET, ShapeT, T, Inds...> &&p,
                       no simplifiable) {
  return static_cast<permute_result<ET, ShapeT, T, Inds...> &&>(p);
}
template <class ET, class ShapeT, class T, size_t... Inds>
constexpr T _simplify_permute_impl(permute_result<ET, ShapeT, T, Inds...> &&p,
                                   yes simplifiable) {
  return std::move(p).input;
}
// _simplify_permute
template <class ET, class ShapeT, class T, size_t... Inds>
constexpr auto
_simplify_permute(permute_result<ET, ShapeT, T, Inds...> &&p) {
  return _simplify_permute_impl(
      std::move(p), (const_ints<size_t, Inds...>() ==
                     make_const_sequence(const_size<sizeof...(Inds)>()))
                        .all());
}
}

// permute
template <class ET, class ShapeT, class T, class... IndexTs>
constexpr auto permute(const tensor_base<ET, ShapeT, T> &t,
                                 const IndexTs &...) {
  static_assert(sizeof...(IndexTs) == ShapeT::rank,
                "invalid number of inds in permute");
  using shape_t = decltype(::wheels::permute(t.shape(), IndexTs()...));
  return details::_simplify_permute(
      permute_result<ET, shape_t, const T &, IndexTs::value...>(t.derived()));
}

template <class ET, class ShapeT, class T, class... IndexTs>
constexpr auto permute(tensor_base<ET, ShapeT, T> &&t,
                                 const IndexTs &...) {
  static_assert(sizeof...(IndexTs) == ShapeT::rank,
                "invalid number of inds in permute");
  using shape_t = decltype(::wheels::permute(t.shape(), IndexTs()...));
  return details::_simplify_permute(
      permute_result<ET, shape_t, T, IndexTs::value...>(
          std::move(t.derived())));
}

// permute a permuted tensor
template <class ET, class ShapeT, class T, size_t... Inds, class... IndexTs>
constexpr auto
permute(const permute_result<ET, ShapeT, T, Inds...> &t, const IndexTs &...) {
  static_assert(sizeof...(Inds) == sizeof...(IndexTs),
                "invalid indices number");
  using shape_t = decltype(::wheels::permute(t.shape(), IndexTs()...));
  return details::_simplify_permute(
      permute_result<ET, shape_t, T, (details::_element<IndexTs::value, size_t,
                                                        Inds...>::value)...>(
          t.input));
}
template <class ET, class ShapeT, class T, size_t... Inds, class... IndexTs>
constexpr auto permute(permute_result<ET, ShapeT, T, Inds...> &&t,
                                 const IndexTs &...) {
  static_assert(sizeof...(Inds) == sizeof...(IndexTs),
                "invalid indices number");
  using shape_t = decltype(::wheels::permute(t.shape(), IndexTs()...));
  return details::_simplify_permute(
      permute_result<ET, shape_t, T, (details::_element<IndexTs::value, size_t,
                                                        Inds...>::value)...>(
          forward<T>(t.input)));
}

// transpose
template <class ST, class MT, class NT, class ET, class T>
constexpr auto
transpose(const tensor_base<ET, tensor_shape<ST, MT, NT>, T> &t) {
  return permute(t.derived(), const_index<1>(), const_index<0>());
}
template <class ST, class MT, class NT, class ET, class T>
auto transpose(tensor_base<ET, tensor_shape<ST, MT, NT>, T> &&t) {
  return permute(std::move(t.derived()), const_index<1>(), const_index<0>());
}
}
