#pragma once

#include "tensor_base_fwd.hpp"

namespace wheels {

template <class ET, class ShapeT, bool StaticShape = ShapeT::is_static>
class iota_result;

template <class ET = size_t, class SizeT> constexpr auto iota(SizeT &&s);

template <class BeginT, class StepT, class EndT>
constexpr decltype(auto) range(BeginT &&b, StepT &&s, EndT &&e);

template <class BeginT, class EndT>
constexpr decltype(auto) range(BeginT &&b, EndT &&e);
}
