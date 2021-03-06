/* * *
 * The MIT License (MIT)
 * 
 * Copyright (c) 2016 Hao Yang (yangh2007@gmail.com)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * * */

#pragma once

#include "tensor_base.hpp"

#include "permute_fwd.hpp"

namespace wheels {

// permute_result
template <class ET, class ShapeT, class T, size_t... Inds>
class permute_result
    : public tensor_base<ET, ShapeT, permute_result<ET, ShapeT, T, Inds...>> {
  static_assert(sizeof...(Inds) == ShapeT::rank,
                "invalid number of inds in permute");

public:
  using value_type = ET;
  using shape_type = ShapeT;
  constexpr explicit permute_result(T &&in) : input(std::forward<T>(in)) {}

public:
  T input;
};

// shape_of
template <class ET, class ShapeT, class T, size_t... Inds>
constexpr auto shape_of(const permute_result<ET, ShapeT, T, Inds...> &m) {
  return permute(m.input.shape(), const_index<Inds>()...);
}

// element_at
namespace detail {
template <class ET, class ShapeT, class T, size_t... Inds, class SubsTupleT,
          size_t... Is>
constexpr decltype(auto)
_element_at_permute_result_seq(const permute_result<ET, ShapeT, T, Inds...> &m,
                               SubsTupleT &&subs, const_ints<size_t, Is...>) {
  return element_at(
      m.input,
      std::get<decltype(::wheels::find_first_of(
          const_ints<size_t, Inds...>(), const_index<Is>()))::value>(subs)...);
}
}
template <class ET, class ShapeT, class T, size_t... Inds, class... SubTs>
constexpr decltype(auto)
element_at(const permute_result<ET, ShapeT, T, Inds...> &m,
           const SubTs &... subs) {
  assert(subscripts_are_valid(m.shape(), subs...));
  return detail::_element_at_permute_result_seq(
      m, std::forward_as_tuple(subs...), make_const_sequence_for<SubTs...>());
}

// unordered
template <class FunT, class ET, class ShapeT, class T, size_t... Inds>
bool for_each_element(behavior_flag<unordered> o, FunT fun,
                      const permute_result<ET, ShapeT, T, Inds...> &m) {
  return for_each_element(o, fun, m.input);
}

// break_on_false
template <class FunT, class ET, class ShapeT, class T, size_t... Inds>
bool for_each_element(behavior_flag<break_on_false> o, FunT fun,
                      const permute_result<ET, ShapeT, T, Inds...> &m) {
  return for_each_element(o, fun, m.input);
}

// nonzero_only
template <class FunT, class ET, class ShapeT, class T, size_t... Inds>
bool for_each_element(behavior_flag<nonzero_only> o, FunT fun,
                      const permute_result<ET, ShapeT, T, Inds...> &m) {
  return for_each_element(o, fun, m.input);
}

// size_t nonzero_elements_count(t)
template <class ET, class ShapeT, class T, size_t... Inds>
inline size_t
nonzero_elements_count(const permute_result<ET, ShapeT, T, Inds...> &t) {
  return nonzero_elements_count(t.input);
}

// reduce_elements
template <class ET, class ShapeT, class T, class E, class ReduceT,
          size_t... Inds>
constexpr E reduce_elements(const permute_result<ET, ShapeT, T, Inds...> &t,
                            E initial, ReduceT &&red) {
  return reduce_elements(t.input, std::move(initial), std::forward<ReduceT>(red));
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

namespace detail {
template <class ET, class ShapeT, class T, size_t... Inds>
constexpr permute_result<ET, ShapeT, T, Inds...>
_simplify_permute_impl(permute_result<ET, ShapeT, T, Inds...> &&p,
                       no simplifiable) {
  return std::move(p);
}
template <class ET, class ShapeT, class T, size_t... Inds>
constexpr T _simplify_permute_impl(permute_result<ET, ShapeT, T, Inds...> &&p,
                                   yes simplifiable) {
  return std::move(p).input;
}
// _simplify_permute
template <class ET, class ShapeT, class T, size_t... Inds>
constexpr decltype(auto)
_simplify_permute(permute_result<ET, ShapeT, T, Inds...> &&p) {
  return _simplify_permute_impl(
      std::move(p), (const_ints<size_t, Inds...>() ==
                     make_const_sequence(const_size<sizeof...(Inds)>()))
                        .all());
}
}

// permute
namespace detail {
template <class ET, class ShapeT, class T, class TT, class... IndexTs>
constexpr decltype(auto) _permute(const tensor_base<ET, ShapeT, T> &, TT &&t,
                                  const IndexTs &...) {
  static_assert(sizeof...(IndexTs) == ShapeT::rank,
                "invalid number of inds in permute");
  using shape_t = decltype(::wheels::permute(t.shape(), IndexTs()...));
  return detail::_simplify_permute(
      permute_result<ET, shape_t, TT, IndexTs::value...>(std::forward<TT>(t)));
}

// permute a permuted tensor
template <class ET, class ShapeT, class T, size_t... Inds, class TT,
          class... IndexTs>
constexpr decltype(auto)
_permute(const permute_result<ET, ShapeT, T, Inds...> &, TT &&t,
         const IndexTs &...) {
  static_assert(sizeof...(Inds) == sizeof...(IndexTs),
                "invalid indices number");
  return _permute(
      t.input, std::forward<TT>(t).input,
      const_index<(
          detail::_element<IndexTs::value, size_t, Inds...>::value)>()...);
}
}
}
