#pragma once

#include "object_fwd.hpp"
#include "overloads.hpp"

namespace wheels {

// const_expr_base
template <class T> struct const_expr_base;

// smart_invoke
template <class FunT, class... ArgTs>
constexpr decltype(auto) smart_invoke(FunT &&fun, ArgTs &&... args);
}
