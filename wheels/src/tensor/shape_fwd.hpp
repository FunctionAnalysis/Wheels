#pragma once

#include "../core/const_ints_fwd.hpp"

namespace wheels {

template <class T, class... SizeTs> class tensor_shape;

template <class T, class SizeT, class... SizeTs>
constexpr auto max_shape_size(const tensor_shape<T, SizeT, SizeTs...> &shape);

template <class T, class SizeT, class... SizeTs>
constexpr auto min_shape_size(const tensor_shape<T, SizeT, SizeTs...> &shape);

template <class T, class... SizeTs>
constexpr auto make_rank_sequence(const tensor_shape<T, SizeTs...> &shape);

template <class... SizeTs> constexpr auto make_shape(const SizeTs &... sizes);

template <class T, class K, class... S1s, class... S2s>
constexpr auto cat2(const tensor_shape<T, S1s...> &t1,
                    const tensor_shape<K, S2s...> &t2);

template <class T, class... Ss, class K, K... Vs>
constexpr auto cat2(const tensor_shape<T, Ss...> &a,
                    const const_ints<K, Vs...> &b);

template <class T, class... Ss, class K, K... Vs>
constexpr auto cat2(const const_ints<K, Vs...> &a,
                    const tensor_shape<T, Ss...> &b);

template <class T, class... Ss, class IntT,
          class = std::enable_if_t<std::is_integral<IntT>::value>>
constexpr auto cat2(const tensor_shape<T, Ss...> &a, const IntT &b);

template <class T, class... Ss, class IntT,
          class = std::enable_if_t<std::is_integral<IntT>::value>>
constexpr auto cat2(const IntT &a, const tensor_shape<T, Ss...> &b);

template <class ShapeOrSizeT, class T, T Times>
constexpr auto repeat_shape(const ShapeOrSizeT &s,
                            const const_ints<T, Times> &times);

template <class T, class... SizeTs, class... IndexTs>
constexpr auto permute(const tensor_shape<T, SizeTs...> &shape,
                       const IndexTs &... inds);
}
