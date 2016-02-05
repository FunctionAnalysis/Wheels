#pragma once

#include "base.hpp"

namespace wheels {

template <class ShapeT, class ET, class IndexTensorT, class InputTensorT>
class index_result : public tensor_op_result_base<
                         ShapeT, ET, void,
                         index_result<ShapeT, ET, IndexTensorT, InputTensorT>> {
public:
  index_result(IndexTensorT &&indt, InputTensorT &&inpt)
      : index_tensor(forward<IndexTensorT>(indt)),
        input_tensor(forward<InputTensorT>(inpt)) {}

public:
  IndexTensorT index_tensor;
  InputTensorT input_tensor;
};

// shape_of
template <class ShapeT, class ET, class IndexTensorT, class InputTensorT>
constexpr decltype(auto)
shape_of(const index_result<ShapeT, ET, IndexTensorT, InputTensorT> &ir) {
  return ir.index_tensor.shape();
}

// element_at
template <class ShapeT, class ET, class IndexTensorT, class InputTensorT,
          class... SubTs>
constexpr decltype(auto)
element_at(const index_result<ShapeT, ET, IndexTensorT, InputTensorT> &ir,
           const SubTs &... subs) {
  return element_at_index(ir.input_tensor,
                          element_at(ir.index_tensor, subs...));
}

// element_at_index
template <class ShapeT, class ET, class IndexTensorT, class InputTensorT,
          class IndexT>
constexpr decltype(auto)
element_at_index(const index_result<ShapeT, ET, IndexTensorT, InputTensorT> &ir,
                 const IndexT &ind) {
  return element_at_index(ir.input_tensor,
                          element_at_index(ir.index_tensor, ind));
}
}