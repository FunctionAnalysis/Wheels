#pragma once

#include "base.hpp"

namespace wheels {

template <class ET, class ShapeT, bool StaticShape = ShapeT::is_static>
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
  return (ET)i;
}

// element_at
template <class ET, class ShapeT, bool StaticShape, class... SubTs>
constexpr ET element_at(const iota_result<ET, ShapeT, StaticShape> &t,
                        const SubTs &... subs) {
  return (ET)sub2ind(t.shape(), subs...);
}

// index_ascending, unordered
template <behavior_flag_enum O, class FunT, class ET, class ShapeT,
          bool StaticShape>
bool for_each_element(behavior_flag<O>, FunT &&fun,
                      const iota_result<ET, ShapeT, StaticShape> &t) {
  for (size_t i = 0; i < numel(t); i++) {
    fun((ET)i);
  }
  return true;
}

// break_on_false
template <class FunT, class ET, class ShapeT, bool StaticShape>
bool for_each_element(behavior_flag<break_on_false>, FunT &&fun,
                      const iota_result<ET, ShapeT, StaticShape> &t) {
  for (size_t i = 0; i < numel(t); i++) {
    if (!fun((ET)i)) {
      return false;
    }
  }
  return true;
}

// nonzero_only
template <class FunT, class ET, class ShapeT, bool StaticShape, class... Ts>
std::enable_if_t<!std::is_scalar<ET>::value, bool>
for_each_element(behavior_flag<nonzero_only> o, FunT &&fun,
                 const iota_result<ET, ShapeT, StaticShape> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 0; i < numel(t); i++) {
    ET e = (ET)i;
    if (!is_zero(e)) {
      fun(e, element_at_index(ts, i)...);
    }
  }
  return true;
}
template <class FunT, class ET, class ShapeT, bool StaticShape, class... Ts>
std::enable_if_t<std::is_scalar<ET>::value, bool>
for_each_element(behavior_flag<nonzero_only> o, FunT &&fun,
                 const iota_result<ET, ShapeT, StaticShape> &t, Ts &&... ts) {
  assert(all_same(shape_of(t), shape_of(ts)...));
  for (size_t i = 1; i < numel(t); i++) {
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
  for (size_t i = 0; i < numel(t); i++) {
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
  for (size_t i = 0; i < numel(t); i++) {
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
template <class ET = size_t, class ST, class... SizeTs>
constexpr auto iota(const tensor_shape<ST, SizeTs...> &s) {
  return iota_result<ET, tensor_shape<ST, SizeTs...>>(s);
}
template <class ET = size_t, class SizeT,
          class = std::enable_if_t<is_int<SizeT>::value>>
constexpr auto iota(const SizeT &s) {
  return iota(make_shape(s));
}
}