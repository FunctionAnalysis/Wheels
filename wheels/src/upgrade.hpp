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

#include "downgrade_fwd.hpp"
#include "upgrade_fwd.hpp"

namespace wheels {

template <class ET, class ShapeT, class InputT, class ExtShapeT, class ExtFunT>
class upgrade_result
    : public tensor_base<
          ET, ShapeT, upgrade_result<ET, ShapeT, InputT, ExtShapeT, ExtFunT>> {
public:
  upgrade_result(InputT &&in, const ExtShapeT &es, ExtFunT ef)
      : input(std::forward<InputT>(in)), ext_shape(es), ext_fun(ef) {}

public:
  InputT input;
  ExtShapeT ext_shape;
  ExtFunT ext_fun;
};

// shape_of
template <class ET, class ShapeT, class InputT, class ExtShapeT, class ExtFunT>
constexpr auto
shape_of(const upgrade_result<ET, ShapeT, InputT, ExtShapeT, ExtFunT> &r) {
  return cat2(r.input.shape(), r.ext_shape);
}

// element_at
namespace detail {
template <class ExtendResultT, class SubsTupleT, size_t... OldIs,
          size_t... ExtendedIs>
constexpr decltype(auto)
_element_at_upgrade_result_seq(ExtendResultT &&r, SubsTupleT &&subs,
                               const const_ints<size_t, OldIs...> &,
                               const const_ints<size_t, ExtendedIs...> &) {
  return r.ext_fun(element_at(r.input, std::get<OldIs>(subs)...),
                   std::get<ExtendedIs + sizeof...(OldIs)>(subs)...);
}
}

template <class ET, class ShapeT, class InputT, class ExtShapeT, class ExtFunT,
          class... SubTs>
constexpr decltype(auto)
element_at(const upgrade_result<ET, ShapeT, InputT, ExtShapeT, ExtFunT> &r,
           const SubTs &... subs) {
  assert(subscripts_are_valid(r.shape(), subs...));
  return detail::_element_at_upgrade_result_seq(
      r, std::forward_as_tuple(subs...), make_rank_sequence(r.input.shape()),
      make_rank_sequence(r.ext_shape));
}

template <class ET, class ShapeT, class InputT, class ExtShapeT, class ExtFunT,
          class... SubTs>
decltype(auto)
element_at(upgrade_result<ET, ShapeT, InputT, ExtShapeT, ExtFunT> &r,
           const SubTs &... subs) {
  assert(subscripts_are_valid(r.shape(), subs...));
  return detail::_element_at_upgrade_result_seq(
      r, std::forward_as_tuple(subs...), make_rank_sequence(r.input.shape()),
      make_rank_sequence(r.ext_shape));
}

namespace upgrade_result_ops {
// as_repeated
struct as_repeated {
  template <class ET, class... SubTs>
  constexpr ET &&operator()(ET &&e, const SubTs &...) const {
    return static_cast<ET &&>(e);
  }
};

// as_subtensor
struct as_subtensor {
  template <class SubTensorT, class... SubTs>
  constexpr decltype(auto) operator()(SubTensorT &&e,
                                      const SubTs &... subs) const {
    return element_at(std::forward<SubTensorT>(e), subs...);
  }
};
}

namespace detail {
template <class ET, class InputShapeT, class InputET, class InputT,
          class InputTT, class ExtShapeT, class ExtFunT, size_t... ExtIs>
constexpr auto _upgrade_by(const tensor_base<InputET, InputShapeT, InputT> &,
                           InputTT &&input, const ExtShapeT &extshape,
                           ExtFunT extfun,
                           const const_ints<size_t, ExtIs...> &) {
  using ele_t = ET;
  using shape_t = std::decay_t<decltype(
      cat2(std::declval<InputShapeT>(), std::declval<ExtShapeT>()))>;
  return upgrade_result<ele_t, shape_t, InputTT, ExtShapeT, ExtFunT>(
      std::forward<InputTT>(input), extshape, extfun);
}
}

// upgrade_as_repeated
namespace detail {
template <class ET, class ShapeT, class T, class InputT, class ST,
          class... SizeTs>
constexpr auto _upgrade_as_repeated(const tensor_base<ET, ShapeT, T> &,
                                    InputT &&input,
                                    const tensor_shape<ST, SizeTs...> &es) {
  return upgrade_by<ET>(std::forward<InputT>(input), es,
                        upgrade_result_ops::as_repeated());
}
}

// upgrade_all
namespace detail {
template <class ET, class ShapeT, class T, class ET2, class ShapeT2, class T2,
          class InputT>
constexpr decltype(auto)
_upgrade_as_subtensor(const tensor_base<ET, ShapeT, T> &et,
                      const tensor_base<ET2, ShapeT2, T2> &, InputT &&input) {
  return upgrade_by<ET>(std::forward<InputT>(input), et.shape(),
                        upgrade_result_ops::as_subtensor());
}
// upgrade a subwised tensor
template <class ET, class ShapeT, class T, class ShapeT2, class T2,
          size_t FixedRank, class InputT>
constexpr decltype(auto)
_upgrade_as_subtensor(const subtensor_view<ET, ShapeT, T, FixedRank> &et,
                      const downgrade_view<ET, ShapeT2, T2, FixedRank> &,
                      InputT &&input) {
  return std::forward<InputT>(input).input;
}
}
template <class InputT>
constexpr decltype(auto) upgrade_all(InputT &&input) {
  assert(input.numel() > 0);
  return detail::_upgrade_as_subtensor(element_at_index(input, 0), input,
                                        std::forward<InputT>(input));
}
}
