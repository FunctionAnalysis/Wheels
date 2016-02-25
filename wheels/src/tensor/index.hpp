#pragma once

#include "base.hpp"
#include "tensor.hpp"

namespace wheels {

template <class ET, class ShapeT, class IndexTensorT, class InputTensorT>
class index_view
    : public tensor_base<ET, ShapeT,
                         index_view<ET, ShapeT, IndexTensorT, InputTensorT>> {
public:
  index_view(IndexTensorT &&indt, InputTensorT &&inpt)
      : index_tensor(forward<IndexTensorT>(indt)),
        input_tensor(forward<InputTensorT>(inpt)) {}

  // operator=
  template <class AnotherT>
  index_view &operator=(const tensor_core<AnotherT> &another) {
    assign_elements(*this, another.derived());
    return *this;
  }
  index_view &operator=(const ET &e) {
    fill_elements_with(*this, e);
    return *this;
  }

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
  return element_at_index(ir.input_tensor,
                          element_at(ir.index_tensor, subs...));
}

// element_at_index
template <class ET, class ShapeT, class IndexTensorT, class InputTensorT,
          class IndexT>
constexpr decltype(auto)
element_at_index(const index_view<ET, ShapeT, IndexTensorT, InputTensorT> &ir,
                 const IndexT &ind) {
  return element_at_index(ir.input_tensor,
                          element_at_index(ir.index_tensor, ind));
}

// at_indices
namespace details {
template <class InputShapeT, class InputET, class InputTensorT,
          class InputTensorTT, class IndexShapeT, class IndexET,
          class IndexTensorT, class IndexTensorTT>
constexpr auto
_at_indices(const tensor_base<InputET, InputShapeT, InputTensorT> &,
            InputTensorTT &&input,
            const tensor_base<IndexET, IndexShapeT, IndexTensorT> &,
            IndexTensorTT &&index) {
  return index_view<InputET, IndexShapeT, IndexTensorTT, InputTensorTT>(
      forward<IndexTensorTT>(index), forward<InputTensorTT>(input));
}
}
template <class InputTensorT, class IndexTensorT>
constexpr auto at_indices(InputTensorT &&input, IndexTensorT &&index)
    -> decltype(details::_at_indices(input, forward<InputTensorT>(input), index,
                                     forward<IndexTensorT>(index))) {
  return details::_at_indices(input, forward<InputTensorT>(input), index,
                              forward<IndexTensorT>(index));
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