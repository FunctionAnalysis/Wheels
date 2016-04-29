#pragma once

namespace wheels {
// smart_invoke
template <class FunT, class... ArgTs>
constexpr decltype(auto) smart_invoke(FunT fun, ArgTs &&... args);
}