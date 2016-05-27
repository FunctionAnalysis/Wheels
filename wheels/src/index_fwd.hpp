#pragma once

#include "tensor_base_fwd.hpp"
#include "tensor_fwd.hpp"

namespace wheels {

template <class ET, class ShapeT, class IndexTensorT, class InputTensorT>
class index_view;

// at_indices
namespace details {
template <class InputShapeT, class InputET, class InputTensorT,
          class InputTensorTT, class IndexShapeT, class IndexET,
          class IndexTensorT, class IndexTensorTT>
constexpr auto
_at_indices(const tensor_base<InputET, InputShapeT, InputTensorT> &,
            InputTensorTT &&input,
            const tensor_base<IndexET, IndexShapeT, IndexTensorT> &,
            IndexTensorTT &&index);
}
template <class InputTensorT, class IndexTensorT>
constexpr auto at_indices(InputTensorT &&input, IndexTensorT &&index)
    -> decltype(details::_at_indices(input, std::forward<InputTensorT>(input),
                                     index,
                                     std::forward<IndexTensorT>(index))) {
  return details::_at_indices(input, std::forward<InputTensorT>(input), index,
                              std::forward<IndexTensorT>(index));
}

// where
template <class ShapeT, class BoolTensorT>
inline vecx_<size_t> where(const tensor_base<bool, ShapeT, BoolTensorT> &flags);
}