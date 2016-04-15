#pragma once

#include "base_fwd.hpp"

namespace wheels {

// auto ptr_of(ts)
template <class T> constexpr auto ptr_of(const tensor_core<T> &t);

// tensor_aligned_data_base
template <class ET, class ShapeT, class T> class tensor_aligned_data_base;

template <class ET, class ShapeT, class T, class... SubTs>
constexpr decltype(auto)
element_at(const tensor_aligned_data_base<ET, ShapeT, T> &t,
           const SubTs &... subs);


// tensor_continuous_data_base
template <class ET, class ShapeT, class T> class tensor_continuous_data_base;

// sub_scale_of
template <class ET, class ShapeT, class T>
constexpr auto sub_scale_of(const tensor_continuous_data_base<ET, ShapeT, T> &);

// sub_offset_of
template <class ET, class ShapeT, class T>
constexpr auto
sub_offset_of(const tensor_continuous_data_base<ET, ShapeT, T> &);

// element_at
template <class ET, class ShapeT, class T, class... SubTs>
constexpr decltype(auto)
element_at(const tensor_continuous_data_base<ET, ShapeT, T> &t,
           const SubTs &... subs);

template <class ET, class ShapeT, class T, class... SubTs>
inline decltype(auto) element_at(tensor_continuous_data_base<ET, ShapeT, T> &t,
                                 const SubTs &... subs);

// element_at_index
template <class ET, class ShapeT, class T, class IndexT>
constexpr decltype(auto)
element_at_index(const tensor_continuous_data_base<ET, ShapeT, T> &t,
                 const IndexT &index);
template <class ET, class ShapeT, class T, class IndexT>
inline decltype(auto)
element_at_index(tensor_continuous_data_base<ET, ShapeT, T> &t,
                 const IndexT &index);

// for_each_element
template <class FunT, class ET, class ShapeT, class T, class... Ts>
bool for_each_element(behavior_flag<index_ascending>, FunT &fun,
                      const tensor_continuous_data_base<ET, ShapeT, T> &t,
                      Ts &... ts);
template <class FunT, class ET, class ShapeT, class T, class... Ts>
bool for_each_element(behavior_flag<index_ascending>, FunT &fun,
                      tensor_continuous_data_base<ET, ShapeT, T> &t,
                      Ts &... ts);

// for_each_element_with_short_circuit
template <class FunT, class ET, class ShapeT, class T, class... Ts>
bool for_each_element(behavior_flag<break_on_false>, FunT &fun,
                      const tensor_continuous_data_base<ET, ShapeT, T> &t,
                      Ts &... ts);

template <class FunT, class ET, class ShapeT, class T, class... Ts>
bool for_each_element(behavior_flag<break_on_false>, FunT &fun,
                      tensor_continuous_data_base<ET, ShapeT, T> &t,
                      Ts &... ts);

// nonzero_only
template <class FunT, class ET, class ShapeT, class T, class... Ts>
bool for_each_element(behavior_flag<nonzero_only>, FunT &fun,
                      const tensor_continuous_data_base<ET, ShapeT, T> &t,
                      Ts &... ts);

template <class FunT, class ET, class ShapeT, class T, class... Ts>
bool for_each_element(behavior_flag<nonzero_only>, FunT &fun,
                      tensor_continuous_data_base<ET, ShapeT, T> &t,
                      Ts &... ts);
}
