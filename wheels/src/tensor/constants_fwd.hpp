#pragma once

#include "base_fwd.hpp"

namespace wheels {

// constant_result
template <class ET, class ShapeT, class OpT> class constant_result;

// shape_of
template <class ET, class ShapeT, class OpT>
constexpr const ShapeT &shape_of(const constant_result<ET, ShapeT, OpT> &t);

// element_at
template <class ET, class ShapeT, class OpT, class... SubTs>
constexpr const ET &element_at(const constant_result<ET, ShapeT, OpT> &t,
                               const SubTs &... subs);

// element_at_index
template <class ET, class ShapeT, class OpT, class IndexT>
constexpr const ET &element_at_index(const constant_result<ET, ShapeT, OpT> &t,
                                     const IndexT &ind);

// index_ascending, unordered
template <behavior_flag_enum O, class FunT, class ET, class ShapeT, class OpT>
bool for_each_element(behavior_flag<O>, FunT &&fun,
                      const constant_result<ET, ShapeT, OpT> &t);

// break_on_false
template <class FunT, class ET, class ShapeT, class OpT>
bool for_each_element(behavior_flag<break_on_false>, FunT &&fun,
                      const constant_result<ET, ShapeT, OpT> &t);

// nonzero_only
template <class FunT, class ET, class ShapeT, class OpT, class... Ts>
bool for_each_element(behavior_flag<nonzero_only> o, FunT &&fun,
                      const constant_result<ET, ShapeT, OpT> &t, Ts &&... ts);

// size_t nonzero_elements_count(t)
template <class ET, class ShapeT, class OpT>
size_t nonzero_elements_count(const constant_result<ET, ShapeT, OpT> &t);

// reduce_elements
template <class ET, class ShapeT, class OpT, class E, class ReduceT>
E reduce_elements(const constant_result<ET, ShapeT, OpT> &t, E initial,
                  ReduceT &&red);

// sum_of
template <class ET, class ShapeT, class OpT>
std::enable_if_t<std::is_scalar<ET>::value, ET>
sum_of(const constant_result<ET, ShapeT, OpT> &t);

// norm_squared
template <class ET, class ShapeT, class OpT>
std::enable_if_t<std::is_scalar<ET>::value, ET>
norm_squared(const constant_result<ET, ShapeT, OpT> &t);

// all_of
template <class ET, class ShapeT, class OpT>
constexpr bool all_of(const constant_result<ET, ShapeT, OpT> &t);

// any_of
template <class ET, class ShapeT, class OpT>
constexpr bool any_of(const constant_result<ET, ShapeT, OpT> &t);

// constants
template <class ET, class ST, class... SizeTs>
constexpr auto constants(const tensor_shape<ST, SizeTs...> &shape, ET &&v);

// zeros
template <class ET = double, class ST, class... SizeTs>
constexpr auto zeros(const tensor_shape<ST, SizeTs...> &shape);
template <class ET = double, class... SizeTs>
constexpr auto zeros(const SizeTs &... sizes);

// ones
template <class ET = double, class ST, class... SizeTs>
constexpr auto ones(const tensor_shape<ST, SizeTs...> &shape);
template <class ET = double, class... SizeTs>
constexpr auto ones(const SizeTs &... sizes);
}