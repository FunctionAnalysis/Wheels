#pragma once

#include "base.hpp"

namespace wheels {

// tensor_aligned_data_base
// - requires: ::wheels::ptr_of, ::wheels::sub_scale_of, ::wheels::sub_offset_of
template <class ShapeT, class ET, class T>
class tensor_aligned_data_base : public tensor_base<ShapeT, ET, T> {
public:
  constexpr const ET *ptr() const { return ::wheels::ptr_of(derived()); }
  ET *ptr() { return ::wheels::ptr_of(derived()); }
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
template <class ShapeT, class ET, class T, size_t... Is, class... SubTs>
constexpr auto
_mem_offset_at_seq(const tensor_aligned_data_base<ShapeT, ET, T> &t,
                   const const_ints<size_t, Is...> &, const SubTs &... subs) {
  return sub2ind(t.shape(), subs * t.sub_scale(const_index<Is>()) +
                                t.sub_offset(const_index<Is>())...);
}
}

// element_at
template <class ShapeT, class ET, class T, class... SubTs>
constexpr const ET &element_at(const tensor_aligned_data_base<ShapeT, ET, T> &t,
                               const SubTs &... subs) {
  return t.ptr()[details::_mem_offset_at_seq(
      t, make_const_sequence_for<SubTs...>(), subs...)];
}
template <class ShapeT, class ET, class T, class... SubTs>
inline ET &element_at(tensor_aligned_data_base<ShapeT, ET, T> &t,
                      const SubTs &... subs) {
  return t.ptr()[details::_mem_offset_at_seq(
      t, make_const_sequence_for<SubTs...>(), subs...)];
}

// element_at_index
template <class ShapeT, class ET, class T, class IndexT>
constexpr const ET &element_at_index(const tensor_aligned_data_base<ShapeT, ET, T> &t,
    const IndexT &index) {

}

}
