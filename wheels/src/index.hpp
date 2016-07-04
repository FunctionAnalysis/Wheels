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
#include "tensor_view_base.hpp"
#include "tensor.hpp"

#include "index_fwd.hpp"

namespace wheels {

template <class ET, class ShapeT, class IndexTensorT, class InputTensorT>
class index_view
    : public tensor_view_base<
          ET, ShapeT, index_view<ET, ShapeT, IndexTensorT, InputTensorT>,
          false> {
  using _base_t = tensor_view_base<
      ET, ShapeT, index_view<ET, ShapeT, IndexTensorT, InputTensorT>, false>;

public:
  index_view(IndexTensorT &&indt, InputTensorT &&inpt)
      : index_tensor(std::forward<IndexTensorT>(indt)),
        input_tensor(std::forward<InputTensorT>(inpt)) {}

  using _base_t::operator=;
  using _base_t::operator+=;
  using _base_t::operator-=;
  using _base_t::operator*=;
  using _base_t::operator/=;

public:
  IndexTensorT index_tensor;
  InputTensorT input_tensor;
};

// shape_of
template <class ET, class ShapeT, class IndexTensorT, class InputTensorT>
constexpr decltype(auto)
shape_of(const index_view<ET, ShapeT, IndexTensorT, InputTensorT> &ir) {
  return ir.index_tensor.shape();
}

// element_at
template <class ET, class ShapeT, class IndexTensorT, class InputTensorT,
          class... SubTs>
constexpr decltype(auto)
element_at(const index_view<ET, ShapeT, IndexTensorT, InputTensorT> &ir,
           const SubTs &... subs) {
  assert(subscripts_are_valid(ir.shape(), subs...));
  return element_at_index(ir.input_tensor,
                          element_at(ir.index_tensor, subs...));
}

// element_at_index
template <class ET, class ShapeT, class IndexTensorT, class InputTensorT,
          class IndexT>
constexpr decltype(auto)
element_at_index(const index_view<ET, ShapeT, IndexTensorT, InputTensorT> &ir,
                 const IndexT &ind) {
  assert(is_between(ind, 0, ir.numel()));
  return element_at_index(ir.input_tensor,
                          element_at_index(ir.index_tensor, ind));
}

// at_indices
namespace detail {
template <class InputShapeT, class InputET, class InputTensorT,
          class InputTensorTT, class IndexShapeT, class IndexET,
          class IndexTensorT, class IndexTensorTT>
constexpr auto
_at_indices(const tensor_base<InputET, InputShapeT, InputTensorT> &,
            InputTensorTT &&input,
            const tensor_base<IndexET, IndexShapeT, IndexTensorT> &,
            IndexTensorTT &&index) {
  return index_view<InputET, IndexShapeT, IndexTensorTT, InputTensorTT>(
      std::forward<IndexTensorTT>(index), std::forward<InputTensorTT>(input));
}
}

// where
template <class ShapeT, class BoolTensorT>
inline vecx_<size_t>
where(const tensor_base<bool, ShapeT, BoolTensorT> &flags) {
  size_t nzc = nonzero_elements_count(flags.derived());
  vecx_<size_t> inds(make_shape(nzc));
  size_t c = 0;
  for (size_t i = 0; i < flags.numel(); i++) {
    if (element_at_index(flags.derived(), i)) {
      element_at_index(inds, c++) = i;
    }
  }
  return inds;
}
}