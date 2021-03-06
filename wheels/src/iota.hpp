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

#include "const_expr.hpp"

#include "tensor_base.hpp"
#include "ewise.hpp"

#include "iota_fwd.hpp"

namespace wheels {

template <class ET, class ShapeT, bool StaticShape>
class iota_result
    : public tensor_base<ET, ShapeT, iota_result<ET, ShapeT, StaticShape>> {
public:
  constexpr explicit iota_result(const ShapeT &s) : _shape(s) {}
  constexpr const ShapeT &shape() const { return _shape; }

private:
  ShapeT _shape;
};
template <class ET, class ShapeT>
class iota_result<ET, ShapeT, true>
    : public tensor_base<ET, ShapeT, iota_result<ET, ShapeT, true>> {
public:
  constexpr explicit iota_result(const ShapeT &) {}
  constexpr ShapeT shape() const { return ShapeT(); }
};

// shape_of
template <class ET, class ShapeT, bool StaticShape>
constexpr decltype(auto)
shape_of(const iota_result<ET, ShapeT, StaticShape> &t) {
  return t.shape();
}

// element_at_index
template <class ET, class ShapeT, bool StaticShape, class IndexT>
constexpr ET element_at_index(const iota_result<ET, ShapeT, StaticShape> &t,
                              const IndexT &i) {
  assert(is_between(i, 0, (typename int_traits<IndexT>::type)t.numel()));
  return (ET)i;
}

// element_at
template <class ET, class ShapeT, bool StaticShape, class... SubTs>
constexpr ET element_at(const iota_result<ET, ShapeT, StaticShape> &t,
                        const SubTs &... subs) {
  assert(subscripts_are_valid(t.shape(), subs...));
  return (ET)sub2ind(t.shape(), subs...);
}

// index_ascending, unordered
template <behavior_flag_enum O, class FunT, class ET, class ShapeT,
          bool StaticShape>
bool for_each_element(behavior_flag<O>, FunT fun,
                      const iota_result<ET, ShapeT, StaticShape> &t) {
  for (size_t i = 0; i < numel_of(t); i++) {
    fun((ET)i);
  }
  return true;
}

// break_on_false
template <class FunT, class ET, class ShapeT, bool StaticShape>
bool for_each_element(behavior_flag<break_on_false>, FunT fun,
                      const iota_result<ET, ShapeT, StaticShape> &t) {
  for (size_t i = 0; i < numel_of(t); i++) {
    if (!fun((ET)i)) {
      return false;
    }
  }
  return true;
}

// nonzero_only
template <class FunT, class ET, class ShapeT, bool StaticShape, class... Ts>
std::enable_if_t<!std::is_scalar<ET>::value, bool>
for_each_element(behavior_flag<nonzero_only> o, FunT fun,
                 const iota_result<ET, ShapeT, StaticShape> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 0; i < numel_of(t); i++) {
    ET e = (ET)i;
    if (!is_zero(e)) {
      fun(e, element_at_index(ts, i)...);
    }
  }
  return true;
}
template <class FunT, class ET, class ShapeT, bool StaticShape, class... Ts>
std::enable_if_t<std::is_scalar<ET>::value, bool>
for_each_element(behavior_flag<nonzero_only> o, FunT fun,
                 const iota_result<ET, ShapeT, StaticShape> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 1; i < numel_of(t); i++) {
    fun((ET)i, element_at_index(ts, i)...);
  }
  return true;
}

// size_t nonzero_elements_count(t)
template <class ET, class ShapeT, bool StaticShape>
std::enable_if_t<std::is_scalar<ET>::value, size_t>
nonzero_elements_count(const iota_result<ET, ShapeT, StaticShape> &t) {
  return t.numel() - 1;
}

template <class ET, class ShapeT, bool StaticShape>
std::enable_if_t<!std::is_scalar<ET>::value, size_t>
nonzero_elements_count(const iota_result<ET, ShapeT, StaticShape> &t) {
  size_t c = 0;
  for (size_t i = 0; i < numel_of(t); i++) {
    ET e = (ET)i;
    if (!is_zero(e)) {
      c++;
    }
  }
  return c;
}

// reduce_elements
template <class ET, class ShapeT, bool StaticShape, class E, class ReduceT>
E reduce_elements(const iota_result<ET, ShapeT, StaticShape> &t, E initial,
                  ReduceT &&red) {
  for (size_t i = 0; i < numel_of(t); i++) {
    initial = red(initial, (ET)i);
  }
  return initial;
}

// norm_squared
template <class ET, class ShapeT, bool StaticShape>
std::enable_if_t<std::is_scalar<ET>::value, ET>
norm_squared(const iota_result<ET, ShapeT, StaticShape> &t) {
  const auto n = t.numel();
  return (n - 1) * n * (2 * n - 1) / 6;
}

// iota
namespace detail {
template <class ET> struct _iota_impl {
  template <class ST, class... SizeTs>
  constexpr auto operator()(const tensor_shape<ST, SizeTs...> &s) const {
    return iota_result<ET, tensor_shape<ST, SizeTs...>>(s);
  }
  constexpr auto operator()(size_t s) const {
    return iota_result<ET, tensor_shape<size_t, size_t>>(
        tensor_shape<size_t, size_t>(s));
  }
  template <class T, T N>
  constexpr auto operator()(const const_ints<T, N> &) const {
    return iota_result<ET, tensor_shape<T, const_ints<T, N>>>(
        tensor_shape<T, const_ints<T, N>>());
  }
};
}
template <class ET, class SizeT> constexpr auto iota(SizeT &&s) {
  return smart_invoke(detail::_iota_impl<ET>(), std::forward<SizeT>(s));
}

// range
namespace detail {
template <class T1, class T2>
constexpr size_t
_range_count(const T1 &t1, const T2 &t2,
             std::enable_if_t<std::is_floating_point<T1>::value ||
                              std::is_floating_point<T2>::value> * = nullptr) {
  assert(t2 != 0);
  return (size_t)conditional(t1 >= 0 != t2 >= 0, 0, std::floor(t1 / t2) + 1.0);
}
template <class T1, class T2>
constexpr size_t _range_count(
    const T1 &t1, const T2 &t2,
    std::enable_if_t<is_int<T1>::value && is_int<T2>::value> * = nullptr) {
  assert(t2 != 0);
  return (size_t)conditional(t1 >= const_int<0>() != t2 >= const_int<0>(), 0,
                             t1 / t2 + 1);
}

struct _range_impl {
  template <class BeginT, class StepT, class EndT>
  constexpr auto operator()(BeginT &&b, StepT &&s, EndT &&e) const {
    using _t =
        std::common_type_t<typename scalar_traits<std::decay_t<BeginT>>::type,
                           typename scalar_traits<std::decay_t<StepT>>::type,
                           typename scalar_traits<std::decay_t<EndT>>::type>;
    return std::forward<BeginT>(b) +
           iota<_t>(_range_count(e - b, s)) * std::forward<StepT>(s);
  }
  template <class BeginT, class EndT>
  constexpr auto operator()(BeginT &&b, EndT &&e) const {
    using _t =
        std::common_type_t<typename scalar_traits<std::decay_t<BeginT>>::type,
                           typename scalar_traits<std::decay_t<EndT>>::type>;
    return std::forward<BeginT>(b) + iota<_t>((size_t)(e - b + const_int<1>()));
  }
};
}
template <class BeginT, class StepT, class EndT>
constexpr decltype(auto) range(BeginT &&b, StepT &&s, EndT &&e) {
  return smart_invoke(detail::_range_impl(), std::forward<BeginT>(b),
                      std::forward<StepT>(s), std::forward<EndT>(e));
}
template <class BeginT, class EndT>
constexpr decltype(auto) range(BeginT &&b, EndT &&e) {
  return smart_invoke(detail::_range_impl(), std::forward<BeginT>(b),
                      std::forward<EndT>(e));
}
}