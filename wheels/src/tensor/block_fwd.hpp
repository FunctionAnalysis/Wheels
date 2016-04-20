#pragma once

#include "base_fwd.hpp"

namespace wheels {

template <class ET, class ShapeT, class InputTensorT,
          class... SubscriptTensorTs>
class block_view;

// at_block
namespace details {
template <class InET, class InShapeT, class InT, class InTT,
          class... SubsTensorTs>
constexpr auto _at_block(const tensor_base<InET, InShapeT, InT> &, InTT &&in,
                         SubsTensorTs &&... sts);
}

template <class InT, class... SubsTensorTs>
constexpr auto at_block(InT &&in, SubsTensorTs &&... sts)
    -> decltype(details::_at_block(in, std::forward<InT>(in),
                                   std::forward<SubsTensorTs>(sts)...)) {
  return details::_at_block(in, std::forward<InT>(in),
                            std::forward<SubsTensorTs>(sts)...);
}
}
