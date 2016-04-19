#pragma once

#include "base_fwd.hpp"

namespace wheels {

// constant_result
template <class ET, class ShapeT, class OpT> class constant_result;

// constants
template <class ET, class ST, class... SizeTs>
constexpr auto constants(const tensor_shape<ST, SizeTs...> &shape, ET &&v);

// zeros
template <class ET = double, class ST, class... SizeTs>
constexpr auto zeros(const tensor_shape<ST, SizeTs...> &shape);
template <class ET = double, class... SizeTs>
constexpr auto zeros(const SizeTs &... sizes);

// ones
template <class ET = double, class ST, class... SizeTs>
constexpr auto ones(const tensor_shape<ST, SizeTs...> &shape);
template <class ET = double, class... SizeTs>
constexpr auto ones(const SizeTs &... sizes);
}