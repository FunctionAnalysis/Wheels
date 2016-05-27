#pragma once

#include "const_ints.hpp"

#include "tensor_base_fwd.hpp"

namespace wheels {

// cat_result
template <class ET, class ShapeT, size_t Axis, class T1, class T2>
class cat_result;

namespace details {
template <size_t Axis, class ShapeT1, class ET1, class T1, class ShapeT2,
          class ET2, class T2, class TT1, class TT2>
constexpr auto _cat_tensor_at(const const_index<Axis> &axis,
                              const tensor_base<ET1, ShapeT1, T1> &,
                              const tensor_base<ET2, ShapeT2, T2> &, TT1 &&in1,
                              TT2 &&in2);
}

template <size_t Axis, class T1, class T2>
constexpr auto cat_at(const const_index<Axis> &axis, T1 &&in1, T2 &&in2)
    -> decltype(details::_cat_tensor_at(axis, in1, in2, std::forward<T1>(in1),
                                        std::forward<T2>(in2))) {
  return details::_cat_tensor_at(axis, in1, in2, std::forward<T1>(in1),
                                 std::forward<T2>(in2));
}

// cat2 (cat_at 0)
template <class T1, class T2>
constexpr auto cat2(T1 &&in1, T2 &&in2)
    -> decltype(cat_at(const_index<0>(), std::forward<T1>(in1),
                       std::forward<T2>(in2))) {
  return cat_at(const_index<0>(), std::forward<T1>(in1), std::forward<T2>(in2));
}

template <class ET, class ShapeT, size_t Axis, class T1, class T2,
          class... SubTs>
constexpr ET element_at(const cat_result<ET, ShapeT, Axis, T1, T2> &m,
                        const SubTs &... subs);

// unordered
template <class FunT, class ET, class ShapeT, size_t Axis, class T1, class T2>
bool for_each_element(behavior_flag<unordered> o, FunT fun,
                      const cat_result<ET, ShapeT, Axis, T1, T2> &t);

// break_on_false
template <class FunT, class ET, class ShapeT, size_t Axis, class T1, class T2>
bool for_each_element(behavior_flag<break_on_false> o, FunT fun,
                      const cat_result<ET, ShapeT, Axis, T1, T2> &t);
// nonzero_only
template <class FunT, class ET, class ShapeT, size_t Axis, class T1, class T2>
bool for_each_element(behavior_flag<nonzero_only> o, FunT fun,
                      const cat_result<ET, ShapeT, Axis, T1, T2> &t);

// size_t nonzero_elements_count(t)
template <class ET, class ShapeT, size_t Axis, class T1, class T2>
inline size_t
nonzero_elements_count(const cat_result<ET, ShapeT, Axis, T1, T2> &t);
}
