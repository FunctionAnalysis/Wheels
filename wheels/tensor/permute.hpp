#pragma once

#include "tensor.hpp"

namespace wheels {

// permute_result
template <class ShapeT, class ET, class T, size_t... Inds>
class permute_result
    : public tensor_op_result_base<ShapeT, ET, void,
                                   permute_result<ShapeT, ET, T, Inds...>> {
  static_assert(sizeof...(Inds) == ShapeT::rank,
                "invalid number of inds in permute");

public:
  using value_type = ET;
  using shape_type = ShapeT;
  constexpr explicit permute_result(T &&in) : _input(forward<T>(in)) {}
  constexpr const T &input() const & { return _input; }
  T &input() & { return _input; }
  T input() && { return std::move(_input); }

private:
  T _input;
};

// shape_of
template <class ShapeT, class ET, class T, size_t... Inds>
constexpr auto shape_of(const permute_result<ShapeT, ET, T, Inds...> &m) {
  return ::wheels::permute(m.input().shape(), const_index<Inds>()...);
}

// element_at
namespace details {
template <class ShapeT, class ET, class T, size_t... Inds, class SubsTupleT,
          size_t... Is>
constexpr decltype(auto)
_element_at_permute_result_seq(const permute_result<ShapeT, ET, T, Inds...> &m,
                               SubsTupleT &&subs, const_ints<size_t, Is...>) {
  return ::wheels::element_at(
      m.input(),
      std::get<decltype(::wheels::find_first_of(
          const_ints<size_t, Inds...>(), const_index<Is>()))::value>(subs)...);
}
}
template <class ShapeT, class ET, class T, size_t... Inds, class... SubTs>
constexpr decltype(auto)
element_at(const permute_result<ShapeT, ET, T, Inds...> &m,
           const SubTs &... subs) {
  return details::_element_at_permute_result_seq(
      m, std::forward_as_tuple(subs...), make_const_sequence_for<SubTs...>());
}

// for_each_element
template <class FunT, class ShapeT, class ET, class T, size_t... Inds>
void for_each_element(order_flag<unordered> o, FunT &&fun,
                      const permute_result<ShapeT, ET, T, Inds...> &m) {
  for_each_element(o, forward<FunT>(fun), m.input());
}

// for_each_element_if
template <class FunT, class ShapeT, class ET, class T, size_t... Inds>
bool for_each_element_with_short_circuit(
    order_flag<unordered> o, FunT &&fun,
    const permute_result<ShapeT, ET, T, Inds...> &m) {
  return for_each_element_with_short_circuit(o, forward<FunT>(fun), m.input());
}

// for_each_nonzero_element
template <class FunT, class ShapeT, class ET, class T, size_t... Inds>
void for_each_nonzero_element(order_flag<unordered> o, FunT &&fun,
                              const permute_result<ShapeT, ET, T, Inds...> &m) {
  for_each_nonzero_element(o, forward<FunT>(fun), m.input());
}

// reduce_elements
template <class ShapeT, class ET, class T, class E, class ReduceT,
          size_t... Inds>
E reduce_elements(const permute_result<ShapeT, ET, T, Inds...> &t, E initial,
                  ReduceT &&red) {
  return reduce_elements(t.input(), move(initial), forward<ReduceT>(red));
}

// norm_squared
template <class ShapeT, class ET, class T, size_t... Inds>
ET norm_squared(const permute_result<ShapeT, ET, T, Inds...> &t) {
  return norm_squared(t.input());
}

// bool all(s)
template <class ShapeT, class ET, class T, size_t... Inds>
constexpr bool all_of(const permute_result<ShapeT, ET, T, Inds...> &t) {
  return all_of(t.input());
}
// bool any(s)
template <class ShapeT, class ET, class T, size_t... Inds>
constexpr bool any_of(const permute_result<ShapeT, ET, T, Inds...> &t) {
  return any_of(t.input());
}

// permute
template <class ShapeT, class ET, class T, class... IndexTs>
constexpr auto permute(const tensor_base<ShapeT, ET, T> &t,
                       const IndexTs &...) {
  static_assert(sizeof...(IndexTs) == ShapeT::rank,
                "invalid number of inds in permute");
  using shape_t = decltype(::wheels::permute(t.shape(), IndexTs()...));
  return permute_result<shape_t, ET, const T &, IndexTs::value...>(t.derived());
}

template <class ShapeT, class ET, class T, class... IndexTs>
constexpr auto permute(tensor_base<ShapeT, ET, T> &&t, const IndexTs &...) {
  static_assert(sizeof...(IndexTs) == ShapeT::rank,
                "invalid number of inds in permute");
  using shape_t = decltype(::wheels::permute(t.shape(), IndexTs()...));
  return permute_result<shape_t, ET, T, IndexTs::value...>(
      std::move(t.derived()));
}

// permute a permuted tensor
namespace details {
template <class ShapeT, class ET, class T, size_t... Inds>
constexpr decltype(auto)
_simplify_impl(permute_result<ShapeT, ET, T, Inds...> &&p, no simplifiable) {
  return static_cast<permute_result<ShapeT, ET, T, Inds...> &&>(p);
}
template <class ShapeT, class ET, class T, size_t... Inds>
constexpr decltype(auto)
_simplify_impl(permute_result<ShapeT, ET, T, Inds...> &&p, yes simplifiable) {
  return std::move(p).input();
}
template <class ShapeT, class ET, class T, size_t... Inds>
constexpr decltype(auto) _simplify(permute_result<ShapeT, ET, T, Inds...> &&p) {
  return _simplify_impl(std::move(p),
                        (const_ints<size_t, Inds...>() ==
                         make_const_sequence(const_size<sizeof...(Inds)>()))
                            .all());
}
}
template <class ShapeT, class ET, class T, size_t... Inds, class... IndexTs>
constexpr auto permute(const permute_result<ShapeT, ET, T, Inds...> &t,
                       const IndexTs &...) {
  static_assert(sizeof...(Inds) == sizeof...(IndexTs),
                "invalid indices number");
  using shape_t = decltype(::wheels::permute(t.shape(), IndexTs()...));
  return details::_simplify(
      permute_result<shape_t, ET, T, (details::_element<IndexTs::value, size_t,
                                                        Inds...>::value)...>(
          t.input()));
}
template <class ShapeT, class ET, class T, size_t... Inds, class... IndexTs>
constexpr auto permute(permute_result<ShapeT, ET, T, Inds...> &&t,
                       const IndexTs &...) {
  static_assert(sizeof...(Inds) == sizeof...(IndexTs),
                "invalid indices number");
  using shape_t = decltype(::wheels::permute(t.shape(), IndexTs()...));
  return details::_simplify(
      permute_result<shape_t, ET, T, (details::_element<IndexTs::value, size_t,
                                                        Inds...>::value)...>(
          move(t).input()));
}

// transpose
template <class ST, class MT, class NT, class ET, class T>
constexpr auto
transpose(const tensor_base<tensor_shape<ST, MT, NT>, ET, T> &t) {
  return permute(t.derived(), const_index<1>(), const_index<0>());
}
template <class ST, class MT, class NT, class ET, class T>
auto transpose(tensor_base<tensor_shape<ST, MT, NT>, ET, T> &&t) {
  return permute(std::move(t.derived()), const_index<1>(), const_index<0>());
}
}
