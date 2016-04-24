#pragma once

#include "base_fwd.hpp"

namespace wheels {

template <class ET, class ShapeT, bool StaticShape = ShapeT::is_static>
class iota_result;

template <class ET = size_t, class SizeT> constexpr auto iota(const SizeT &s);

template <class BeginT, class StepT, class EndT>
constexpr decltype(auto) range(const BeginT &b, const StepT &s, const EndT &e);

template <class BeginT, class EndT>
constexpr decltype(auto) range(const BeginT &b, const EndT &e);
}
