#pragma once

#include "object_fwd.hpp"

namespace wheels {

template <class OpT> struct func_base;

template <class OpT, class... CatTs>
inline void overload_as(const func_base<OpT> &,
                        const category::other<CatTs> &...);
}
