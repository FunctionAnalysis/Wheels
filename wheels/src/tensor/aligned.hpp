#pragma once

#include "aligned_fwd.hpp"

namespace wheels {

template <class T> constexpr auto ptr_of(const tensor_core<T> &t) {
  static_assert(always<bool, false, T>::value,
                "ptr_of(const T &) is not supported by tensor_core<T>, do "
                "you forget to call .derived()?");
}

// tensor_aligned_data_base
// - requires: ::wheels::ptr_of, ::wheels::sub_scale_of, ::wheels::sub_offset_of
template <class ET, class ShapeT, class T>
class tensor_aligned_data_base : public tensor_base<ET, ShapeT, T> {
public:
  constexpr auto ptr() const { return ::wheels::ptr_of(derived()); }
  auto ptr() { return ::wheels::ptr_of(derived()); }
  template <size_t Idx>
  constexpr auto sub_scale(const const_index<Idx> &i) const {
    return ::wheels::sub_scale_of(derived(), i);
  }
  template <size_t Idx>
  constexpr auto sub_offset(const const_index<Idx> &i) const {
    return ::wheels::sub_offset_of(derived(), i);
  }
};

namespace details {
template <class ET, class ShapeT, class T, size_t... Is, class... SubTs>
constexpr auto
_mem_offset_at_seq(const tensor_aligned_data_base<ET, ShapeT, T> &t,
                   const const_ints<size_t, Is...> &, const SubTs &... subs) {
  return sub2ind(t.shape(), subs * t.sub_scale(const_index<Is>()) +
                                t.sub_offset(const_index<Is>())...);
}
}

// element_at
template <class ET, class ShapeT, class T, class... SubTs>
constexpr decltype(auto)
element_at(const tensor_aligned_data_base<ET, ShapeT, T> &t,
           const SubTs &... subs) {
  return t.ptr()[details::_mem_offset_at_seq(
      t, make_const_sequence_for<SubTs...>(), subs...)];
}
template <class ET, class ShapeT, class T, class... SubTs>
inline decltype(auto) element_at(tensor_aligned_data_base<ET, ShapeT, T> &t,
                                 const SubTs &... subs) {
  return t.ptr()[details::_mem_offset_at_seq(
      t, make_const_sequence_for<SubTs...>(), subs...)];
}

// tensor_continuous_data_base
template <class ET, class ShapeT, class T>
class tensor_continuous_data_base
    : public tensor_aligned_data_base<ET, ShapeT, T> {};

// sub_scale_of
template <class ET, class ShapeT, class T>
constexpr auto
sub_scale_of(const tensor_continuous_data_base<ET, ShapeT, T> &) {
  return const_size<1>();
}

// sub_offset_of
template <class ET, class ShapeT, class T>
constexpr auto
sub_offset_of(const tensor_continuous_data_base<ET, ShapeT, T> &) {
  return const_size<0>();
}

// element_at
template <class ET, class ShapeT, class T, class... SubTs>
constexpr decltype(auto)
element_at(const tensor_continuous_data_base<ET, ShapeT, T> &t,
           const SubTs &... subs) {
  return t.ptr()[sub2ind(t.shape(), subs...)];
}
template <class ET, class ShapeT, class T, class... SubTs>
inline decltype(auto) element_at(tensor_continuous_data_base<ET, ShapeT, T> &t,
                                 const SubTs &... subs) {
  return t.ptr()[sub2ind(t.shape(), subs...)];
}

// element_at_index
template <class ET, class ShapeT, class T, class IndexT>
constexpr decltype(auto)
element_at_index(const tensor_continuous_data_base<ET, ShapeT, T> &t,
                 const IndexT &index) {
  return t.ptr()[index];
}
template <class ET, class ShapeT, class T, class IndexT>
inline decltype(auto)
element_at_index(tensor_continuous_data_base<ET, ShapeT, T> &t,
                 const IndexT &index) {
  return t.ptr()[index];
}

// for_each_element
template <class FunT, class ET, class ShapeT, class T, class... Ts>
bool for_each_element(behavior_flag<index_ascending>, FunT &fun,
                      const tensor_continuous_data_base<ET, ShapeT, T> &t,
                      Ts &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  for (size_t i = 0; i < t.numel(); i++) {
    fun(element_at_index(t.derived(), i), element_at_index(ts.derived(), i)...);
  }
  return true;
}
template <class FunT, class ET, class ShapeT, class T, class... Ts>
bool for_each_element(behavior_flag<index_ascending>, FunT &fun,
                      tensor_continuous_data_base<ET, ShapeT, T> &t,
                      Ts &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  for (size_t i = 0; i < t.numel(); i++) {
    fun(element_at_index(t.derived(), i), element_at_index(ts.derived(), i)...);
  }
  return true;
}

// for_each_element_with_short_circuit
template <class FunT, class ET, class ShapeT, class T, class... Ts>
bool for_each_element(behavior_flag<break_on_false>, FunT &fun,
                      const tensor_continuous_data_base<ET, ShapeT, T> &t,
                      Ts &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  for (size_t i = 0; i < t.numel(); i++) {
    bool r = fun(element_at_index(t.derived(), i),
                 element_at_index(ts.derived(), i)...);
    if (!r) {
      return false;
    }
  }
  return true;
}

template <class FunT, class ET, class ShapeT, class T, class... Ts>
bool for_each_element(behavior_flag<break_on_false>, FunT &fun,
                      tensor_continuous_data_base<ET, ShapeT, T> &t,
                      Ts &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  for (size_t i = 0; i < t.numel(); i++) {
    bool r = fun(element_at_index(t.derived(), i),
                 element_at_index(ts.derived(), i)...);
    if (!r) {
      return false;
    }
  }
  return true;
}

// nonzero_only
template <class FunT, class ET, class ShapeT, class T, class... Ts>
bool for_each_element(behavior_flag<nonzero_only>, FunT &fun,
                      const tensor_continuous_data_base<ET, ShapeT, T> &t,
                      Ts &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  bool visited_all = true;
  for (size_t i = 0; i < t.numel(); i++) {
    decltype(auto) e = element_at_index(t.derived(), i);
    if (e) {
      fun(e, element_at_index(ts.derived(), i)...);
    } else {
      visited_all = false;
    }
  }
  return visited_all;
}

template <class FunT, class ET, class ShapeT, class T, class... Ts>
bool for_each_element(behavior_flag<nonzero_only>, FunT &fun,
                      tensor_continuous_data_base<ET, ShapeT, T> &t,
                      Ts &... ts) {
  assert(all_same(t.shape(), ts.shape()...));
  bool visited_all = true;
  for (size_t i = 0; i < t.numel(); i++) {
    decltype(auto) e = element_at_index(t.derived(), i);
    if (e) {
      fun(e, element_at_index(ts.derived(), i)...);
    } else {
      visited_all = false;
    }
  }
  return visited_all;
}
}
