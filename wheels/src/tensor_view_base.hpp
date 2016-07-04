#pragma once

#include "aligned.hpp"
#include "ewise.hpp"
#include "tensor_base.hpp"

namespace wheels {
namespace detail {
template <class ET, class ShapeT, class T, bool IsContinuousData>
struct _select_tensor_base;

template <class ET, class ShapeT, class T>
struct _select_tensor_base<ET, ShapeT, T, false> {
  using type = tensor_base<ET, ShapeT, T>;
};
template <class ET, class ShapeT, class T>
struct _select_tensor_base<ET, ShapeT, T, true> {
  using type = tensor_continuous_data_base<ET, ShapeT, T>;
};
template <class ET, class ShapeT, class T, bool IsContinuousData>
using _select_tensor_base_t =
    typename _select_tensor_base<ET, ShapeT, T, IsContinuousData>::type;
}

template <class ET, class ShapeT, class T, bool IsContinuousData = false>
class tensor_view_base
    : public detail::_select_tensor_base_t<ET, ShapeT, T, IsContinuousData> {
public:
  // operator=
  template <class AnotherShapeT, class AnotherT,
            class = std::enable_if_t<AnotherShapeT::rank == ShapeT::rank>>
  T &operator=(const tensor_base<ET, AnotherShapeT, AnotherT> &another) {
    assign_elements(this->derived(), another.derived());
    return this->derived();
  }
  T &operator=(const ET &e) {
    fill_elements_with(this->derived(), e);
    return this->derived();
  }
  template <class AnotherET, class AnotherShapeT, class AnotherT>
  T &operator=(const scalarize_wrapper<AnotherET, AnotherShapeT, AnotherT> &t) {
    fill_elements_with(this->derived(), t.host);
    return this->derived();
  }

  // +=
  template <class AnotherShapeT, class AnotherT,
            class = std::enable_if_t<AnotherShapeT::rank == ShapeT::rank>>
  T &operator+=(const tensor_base<ET, AnotherShapeT, AnotherT> &t) {
    assert(this->shape() == t.shape());
    for_each_element(behavior_flag<unordered>(),
                     [](auto &&ele1, auto &&ele2) { ele1 += ele2; },
                     this->derived(), t.derived());
    return this->derived();
  }
  T &operator+=(const ET &e) {
    for_each_element(behavior_flag<unordered>(), [&e](auto &&ele) { ele += e; },
                     this->derived());
    return this->derived();
  }
  template <class AnotherET, class AnotherShapeT, class AnotherT>
  T &operator+=(
      const scalarize_wrapper<AnotherET, AnotherShapeT, AnotherT> &t) {
    for_each_element(behavior_flag<unordered>(),
                     [&t](auto &&ele) { ele += t.host; }, this->derived());
    return this->derived();
  }

  // -=
  template <class AnotherShapeT, class AnotherT,
            class = std::enable_if_t<AnotherShapeT::rank == ShapeT::rank>>
  T &operator-=(const tensor_base<ET, AnotherShapeT, AnotherT> &t) {
    assert(this->shape() == t.shape());
    for_each_element(behavior_flag<unordered>(),
                     [](auto &&ele1, auto &&ele2) { ele1 -= ele2; },
                     this->derived(), t.derived());
    return this->derived();
  }
  T &operator-=(const ET &e) {
    for_each_element(behavior_flag<unordered>(), [&e](auto &&ele) { ele -= e; },
                     this->derived());
    return this->derived();
  }
  template <class AnotherET, class AnotherShapeT, class AnotherT>
  T &operator-=(
      const scalarize_wrapper<AnotherET, AnotherShapeT, AnotherT> &t) {
    for_each_element(behavior_flag<unordered>(),
                     [&t](auto &&ele) { ele -= t.host; }, this->derived());
    return this->derived();
  }

  // *=
  T &operator*=(const ET &e) {
    for_each_element(behavior_flag<unordered>(), [&e](auto &&ele) { ele *= e; },
                     this->derived());
    return this->derived();
  }

  // /=
  T &operator/=(const ET &e) {
    for_each_element(behavior_flag<unordered>(), [&e](auto &&ele) { ele /= e; },
                     this->derived());
    return this->derived();
  }
};
}