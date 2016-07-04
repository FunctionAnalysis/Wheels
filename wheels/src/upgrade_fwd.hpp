#pragma once

#include "tensor_base_fwd.hpp"
#include "shape_fwd.hpp"

namespace wheels {

template <class ET, class ShapeT, class InputT, class ExtShapeT, class ExtFunT>
class upgrade_result;

namespace detail {
template <class ET, class InputShapeT, class InputET, class InputT,
          class InputTT, class ExtShapeT, class ExtFunT, size_t... ExtIs>
constexpr auto _upgrade_by(const tensor_base<InputET, InputShapeT, InputT> &,
                           InputTT &&input, const ExtShapeT &extshape,
                           ExtFunT extfun,
                           const const_ints<size_t, ExtIs...> &);
}

// upgrade_by
template <class ET, class InputT, class ST, class... SizeTs, class ExtFunT>
constexpr auto upgrade_by(InputT &&input, const tensor_shape<ST, SizeTs...> &es,
                          ExtFunT ef)
    -> decltype(detail::_upgrade_by<ET>(input, std::forward<InputT>(input), es,
                                         ef, make_rank_sequence(es))) {
  return detail::_upgrade_by<ET>(input, std::forward<InputT>(input), es, ef,
                                  make_rank_sequence(es));
}

// upgrade_as_repeated
namespace detail {
template <class ET, class ShapeT, class T, class InputT, class ST,
          class... SizeTs>
constexpr auto _upgrade_as_repeated(const tensor_base<ET, ShapeT, T> &,
                                    InputT &&input,
                                    const tensor_shape<ST, SizeTs...> &es);
}

template <class InputT, class ST, class... SizeTs>
constexpr auto upgrade_as_repeated(InputT &&input,
                                   const tensor_shape<ST, SizeTs...> &es) {
  return detail::_upgrade_as_repeated(input, std::forward<InputT>(input), es);
}

// upgrade_all
template <class InputT>
constexpr decltype(auto) upgrade_all(InputT &&input);
}