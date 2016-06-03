#pragma once

#include "tensor_base_fwd.hpp"

namespace wheels {
template <class ET, class ShapeT> class tensor_map;

template <class E, class ST, class... SizeTs>
constexpr auto map(const tensor_shape<ST, SizeTs...> &shape, E *mem);
template <class E, size_t N> constexpr auto map(E (&arr)[N]);

namespace literals {
inline auto operator"" _ts(const char *str, size_t s);
inline auto operator"" _ts(const wchar_t *str, size_t s);
inline auto operator"" _ts(const char16_t *str, size_t s);
inline auto operator"" _ts(const char32_t *str, size_t s);
}
}
