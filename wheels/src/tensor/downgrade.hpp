#pragma once

#include "base.hpp"
#include "tensor.hpp"

#include "downgrade_fwd.hpp"
#include "upgrade_fwd.hpp"

namespace wheels {

namespace details {
template <class T1, class T2>
struct _is_same_intrinsic : std::is_same<std::decay_t<T1>, std::decay_t<T2>> {};
}

// subtensor_view
template <class ET, class SubShapeT, class InputT, size_t FixedRank>
class subtensor_view
    : public tensor_base<ET, SubShapeT,
                         subtensor_view<ET, SubShapeT, InputT, FixedRank>> {
public:
  template <class... SubTs>
  constexpr subtensor_view(InputT &&in, const SubTs &... subs)
      : input(std::forward<InputT>(in)), fixed_subs{{(size_t)subs...}} {}

  // operator=
  template <class AnotherT>
  subtensor_view &operator=(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
    return *this;
  }
  //template <class InputT2, class = std::enable_if_t<details::_is_same_intrinsic<
  //                             InputT, InputT2>::value>>
  //subtensor_view &
  //operator=(const subtensor_view<ET, SubShapeT, InputT2, FixedRank> &t) {
  //  input = t.input;
  //  fixed_subs = t.fixed_subs;
  //  return *this;
  //}

public:
  InputT input;
  std::array<size_t, FixedRank> fixed_subs;
};

// shape_of
template <class ET, class ShapeT, class InputT, size_t FixedRank>
constexpr auto
shape_of(const subtensor_view<ET, ShapeT, InputT, FixedRank> &b) {
  return b.input.shape().part(make_const_range(
      const_index<FixedRank>(), const_index<FixedRank + ShapeT::rank>()));
}

// element_at
namespace details {
template <class SubTensorViewT, size_t... Is, class... SubTs>
constexpr decltype(auto)
_element_at_subtensor_view_seq(SubTensorViewT &b,
                               const const_ints<size_t, Is...> &,
                               const SubTs &... subs) {
  return element_at(b.input, std::get<Is>(b.fixed_subs)..., subs...);
}
}
template <class ET, class ShapeT, class InputT, size_t FixedRank,
          class... SubTs>
constexpr decltype(auto)
element_at(const subtensor_view<ET, ShapeT, InputT, FixedRank> &b,
           const SubTs &... subs) {
  return details::_element_at_subtensor_view_seq(
      b, make_const_sequence(const_size<FixedRank>()), subs...);
}
template <class ET, class ShapeT, class InputT, size_t FixedRank,
          class... SubTs>
decltype(auto) element_at(subtensor_view<ET, ShapeT, InputT, FixedRank> &b,
                          const SubTs &... subs) {
  return details::_element_at_subtensor_view_seq(
      b, make_const_sequence(const_size<FixedRank>()), subs...);
}

// subtensor_at
namespace details {
// _cat_shape
template <class ShapeT1, class ShapeT2> struct _cat_shape {};
template <class T1, class... SizeT1s, class T2, class... SizeT2s>
struct _cat_shape<tensor_shape<T1, SizeT1s...>, tensor_shape<T2, SizeT2s...>> {
  using type = tensor_shape<std::common_type_t<T1, T2>, SizeT1s..., SizeT2s...>;
};
template <class T, class... SizeTs, class SizeT>
struct _cat_shape<tensor_shape<T, SizeTs...>, SizeT> {
  using type = tensor_shape<T, SizeTs..., SizeT>;
};
template <class T, class... SizeTs, class SizeT>
struct _cat_shape<SizeT, tensor_shape<T, SizeTs...>> {
  using type = tensor_shape<T, SizeT, SizeTs...>;
};

// _split_shape
template <class ShapeT, size_t N> struct _split_shape {};
template <class T, class... SizeTs>
struct _split_shape<tensor_shape<T, SizeTs...>, (size_t)0> {
  using head = tensor_shape<T>;
  using tail = tensor_shape<T, SizeTs...>;
};
template <class T, class SizeT, class... SizeTs>
struct _split_shape<tensor_shape<T, SizeT, SizeTs...>, (size_t)0> {
  using head = tensor_shape<T>;
  using tail = tensor_shape<T, SizeT, SizeTs...>;
};
template <class T, class SizeT, class... SizeTs, size_t N>
struct _split_shape<tensor_shape<T, SizeT, SizeTs...>, N> {
  using head = typename _cat_shape<
      SizeT,
      typename _split_shape<tensor_shape<T, SizeTs...>, N - 1>::head>::type;
  using tail = typename _split_shape<tensor_shape<T, SizeTs...>, N - 1>::tail;
};

template <class ShapeT, size_t N>
using _head_of_shape_t = typename _split_shape<ShapeT, N>::head;
template <class ShapeT, size_t N>
using _tail_of_shape_t = typename _split_shape<ShapeT, N>::tail;

template <class ET, class ShapeT, class T, class TT, class... SubTs>
constexpr auto _subtensor_at(const tensor_base<ET, ShapeT, T> &, TT &&input,
                             const SubTs &... subs) {
  static_assert(sizeof...(SubTs) < ShapeT::rank, "two many subscripts");
  using shape_t = _tail_of_shape_t<ShapeT, sizeof...(SubTs)>;
  return subtensor_view<ET, shape_t, TT, sizeof...(SubTs)>(
      std::forward<TT>(input), subs...);
}
}

// downgrade_view
template <class ET, class ShapeT, class InputT, size_t FixedRank>
class downgrade_view
    : public tensor_base<
          tensor<ET, details::_tail_of_shape_t<ShapeT, FixedRank>>,
          details::_head_of_shape_t<ShapeT, FixedRank>,
          downgrade_view<ET, ShapeT, InputT, FixedRank>> {
  static_assert(FixedRank <= ShapeT::rank, "fixed rank overflow");

public:
  constexpr explicit downgrade_view(InputT &&in)
      : input(std::forward<InputT>(in)) {}

  // operator=
  template <class AnotherT>
  downgrade_view &operator=(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
    return *this;
  }

public:
  InputT input;
};

// shape_of
template <class ET, class ShapeT, class InputT, size_t FixedRank>
constexpr auto
shape_of(const downgrade_view<ET, ShapeT, InputT, FixedRank> &t) {
  return t.input.shape().part(
      make_const_range(const_index<0>(), const_index<FixedRank>()));
}

// element_at
namespace details {
template <class SubTensorViewT, class SubwiseViewT, size_t... Is,
          class SubsTupleT>
constexpr SubTensorViewT _element_at_downgrade_view_seq(
    SubwiseViewT &bv, const const_ints<size_t, Is...> &, SubsTupleT &&subs) {
  return SubTensorViewT(bv.input, std::get<Is>(subs)...);
}
}

template <class ET, class ShapeT, class InputT, size_t FixedRank,
          class... SubTs>
constexpr auto
element_at(const downgrade_view<ET, ShapeT, InputT, FixedRank> &t,
           const SubTs &... subs) {
  using const_subtensor_view_view_t =
      subtensor_view<ET, details::_tail_of_shape_t<ShapeT, FixedRank>,
                     const InputT &, FixedRank>;
  return details::_element_at_downgrade_view_seq<const_subtensor_view_view_t>(
      t, make_const_sequence(const_size<FixedRank>()),
      std::forward_as_tuple(subs...));
}

template <class ET, class ShapeT, class InputT, size_t FixedRank,
          class... SubTs>
auto element_at(downgrade_view<ET, ShapeT, InputT, FixedRank> &t,
                const SubTs &... subs) {
  using subtensor_view_view_t =
      subtensor_view<ET, details::_tail_of_shape_t<ShapeT, FixedRank>, InputT &,
                     FixedRank>;
  return details::_element_at_downgrade_view_seq<subtensor_view_view_t>(
      t, make_const_sequence(const_size<FixedRank>()),
      std::forward_as_tuple(subs...));
}

// downgrade
namespace details {
template <class ET, class ShapeT, class InputT, class TT, size_t FixedRank>
constexpr auto _downgrade(const tensor_base<ET, ShapeT, InputT> &, TT &&input,
                          const const_size<FixedRank> &) {
  return downgrade_view<ET, ShapeT, TT, FixedRank>(std::forward<TT>(input));
}
// downgrade an upgraded tensor
template <class ET, class ShapeT, class InputT, class ExtShapeT, class TT>
constexpr decltype(auto)
_downgrade(const upgrade_result<ET, ShapeT, InputT, ExtShapeT,
                                upgrade_result_ops::as_subtensor> &,
           TT &&input, const const_size<ExtShapeT::rank> &) {
  return std::forward<TT>(input).input;
}
}
}
