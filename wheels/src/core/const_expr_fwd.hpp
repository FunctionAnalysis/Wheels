#pragma once

#include "object_fwd.hpp"
#include "overloads.hpp"

namespace wheels {

// const_expr_base
template <class T> struct const_expr_base;

namespace literals {
// ""_symbol
template <char... Cs> constexpr auto operator"" _symbol();
}
}
