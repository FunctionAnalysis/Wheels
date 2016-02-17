#pragma once

#include "base.hpp"

namespace wheels {

template <class ShapeT, class ET, class IndexTensorT, class InputTensorT>
class index_view : public tensor_op_result_base<
                       ShapeT, ET, void,
                       index_view<ShapeT, ET, IndexTensorT, InputTensorT>> {
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

public:
  IndexTensorT index_tensor;
  InputTensorT input_tensor;
};

// shape_of
template <class ShapeT, class ET, class IndexTensorT, class InputTensorT>
constexpr decltype(auto)
shape_of(const index_view<ShapeT, ET, IndexTensorT, InputTensorT> &ir) {
  return ir.index_tensor.shape();
}

// element_at
template <class ShapeT, class ET, class IndexTensorT, class InputTensorT,
          class... SubTs>
constexpr decltype(auto)
element_at(const index_view<ShapeT, ET, IndexTensorT, InputTensorT> &ir,
           const SubTs &... subs) {
  return element_at_index(ir.input_tensor,
                          element_at(ir.index_tensor, subs...));
}

// element_at_index
template <class ShapeT, class ET, class IndexTensorT, class InputTensorT,
          class IndexT>
constexpr decltype(auto)
element_at_index(const index_view<ShapeT, ET, IndexTensorT, InputTensorT> &ir,
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
_at_indices(const tensor_base<InputShapeT, InputET, InputTensorT> &,
            InputTensorTT &&input,
            const tensor_base<IndexShapeT, IndexET, IndexTensorT> &,
            IndexTensorTT &&index) {
  return index_view<IndexShapeT, InputET, IndexTensorTT, InputTensorTT>(
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
}