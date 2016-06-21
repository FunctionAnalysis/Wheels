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

#include "reformulate_fwd.hpp"

namespace wheels {

// reformulate_result
template <class ET, class ShapeT, class InputT, class SubsMapFunT>
class reformulate_result
    : public tensor_base<ET, ShapeT,
                         reformulate_result<ET, ShapeT, InputT, SubsMapFunT>> {
public:
  constexpr explicit reformulate_result(const ShapeT &s, SubsMapFunT f,
                                        InputT &&in)
      : _shape(s), _subs_map_fun(f), _input(std::forward<InputT>(in)) {}

  constexpr const ShapeT &shape() const { return _shape; }
  constexpr const InputT &input() const & { return _input; }
  InputT &input() & { return _input; }
  InputT &&input() && { return _input; }

  template <class... SubTs>
  constexpr decltype(auto) subs_of_input(const SubTs &... subs) const {
    return _subs_map_fun(subs...);
  }

private:
  ShapeT _shape;
  SubsMapFunT _subs_map_fun;
  InputT _input;
};

// shape_of
template <class ET, class ShapeT, class InputT, class SubsMapFunT>
constexpr const ShapeT &
shape_of(const reformulate_result<ET, ShapeT, InputT, SubsMapFunT> &r) {
  return r.shape();
}

// element_at
namespace details {
template <class InputT, class SubsTupleT, size_t... Is>
constexpr decltype(auto)
_element_at_subs_seq(InputT &&input, SubsTupleT &&subs,
                     const const_ints<size_t, Is...> &) {
  return element_at(input, std::get<Is>(subs)...);
}
}
template <class ET, class ShapeT, class InputT, class SubsMapFunT,
          class... SubTs>
constexpr decltype(auto)
element_at(const reformulate_result<ET, ShapeT, InputT, SubsMapFunT> &r,
           const SubTs &... subs) {
  return details::_element_at_subs_seq(r.input(), r.subs_of_input(subs...),
                                       make_rank_sequence(r.input().shape()));
}

namespace details {
template <class ET, class ShapeT, class T, class TT, class NewShapeT,
          class SubsMapFunT>
constexpr auto _reformulate(const tensor_base<ET, ShapeT, T> &, TT &&t,
                            const NewShapeT &s, SubsMapFunT fun) {
  return reformulate_result<ET, NewShapeT, TT, SubsMapFunT>(
      s, fun, std::forward<TT>(t));
}
}

// repeat
namespace details {
template <class ShapeT, class SubsTupleT, size_t... Is>
constexpr auto _repeat_subs_seq(const ShapeT &shape, SubsTupleT &&subs,
                                const const_ints<size_t, Is...> &) {
  return std::make_tuple((std::get<Is>(subs) % std::get<Is>(shape))...);
}
template <class ShapeT> struct _repeat_subs_functor {
  ShapeT shape;
  constexpr explicit _repeat_subs_functor(const ShapeT &s) : shape(s) {}
  template <class... SubTs>
  constexpr auto operator()(const SubTs &... subs) const {
    return _repeat_subs_seq(shape, std::make_tuple(subs...),
                            make_const_sequence_for<SubTs...>());
  }
};

template <class ET, class ShapeT, class T, class TT, class RepsTupleT,
          size_t... Is>
constexpr auto _repeat_impl(const tensor_base<ET, ShapeT, T> &, TT &&t,
                            RepsTupleT &&rps,
                            const const_ints<size_t, Is...> &) {
  static_assert(sizeof...(Is) == ShapeT::rank, "wrong number of repeats");
  return _reformulate(
      t, std::forward<T>(t),
      make_shape((std::get<Is>(rps) * t.size(const_index<Is>()))...),
      _repeat_subs_functor<ShapeT>(t.shape()));
}
}
}