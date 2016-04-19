#pragma once

#include "shape_fwd.hpp"
#include "base_fwd.hpp"

namespace wheels {
template <class ET, class ShapeT> class tensor;

// vec_
template <class T, size_t N>
using vec_ = tensor<T, tensor_shape<size_t, const_size<N>>>;
using vec2 = vec_<double, 2>;
using vec3 = vec_<double, 3>;

// gvec_
template <class E, size_t N, class T>
using gvec_ = tensor_base<E, tensor_shape<size_t, const_size<N>>, T>;
template <class T> using gvec2 = gvec_<double, 2, T>;
template <class T> using gvec3 = gvec_<double, 3, T>;

// vecx_
template <class T> using vecx_ = tensor<T, tensor_shape<size_t, size_t>>;
using vecx = vecx_<double>;
using vecxi = vecx_<int>;
using vecxb = vecx_<bool>;

// gvecx_
template <class E, class T>
using gvecx_ = tensor_base<E, tensor_shape<size_t, size_t>, T>;
template <class T> using gvecx = gvecx_<double, T>;

// mat_
template <class T, size_t M, size_t N>
using mat_ = tensor<T, tensor_shape<size_t, const_size<M>, const_size<N>>>;
using mat2 = mat_<double, 2, 2>;
using mat3 = mat_<double, 3, 3>;

// gmat_
template <class E, size_t M, size_t N, class T>
using gmat_ =
    tensor_base<E, tensor_shape<size_t, const_size<M>, const_size<N>>, T>;
template <class T> using gmat2 = gmat_<double, 2, 2, T>;
template <class T> using gmat3 = gmat_<double, 3, 3, T>;

// matx_
template <class T>
using matx_ = tensor<T, tensor_shape<size_t, size_t, size_t>>;
using matx = matx_<double>;

// gmatx_
template <class E, class T>
using gmatx_ = tensor_base<E, tensor_shape<size_t, size_t, size_t>, T>;
template <class T> using gmatx = gmatx_<double, T>;

// rowvec_, colvec_
template <class T, size_t N>
using rowvec_ = tensor<T, tensor_shape<size_t, const_size<1>, const_size<N>>>;
template <class T, size_t N>
using colvec_ = tensor<T, tensor_shape<size_t, const_size<N>, const_size<1>>>;

// rowvecx_, colvecx_
template <class T>
using rowvecx_ = tensor<T, tensor_shape<size_t, const_size<1>, size_t>>;
template <class T>
using colvecx_ = tensor<T, tensor_shape<size_t, size_t, const_size<1>>>;

using rowvecx = rowvecx_<double>;
using colvecx = colvecx_<double>;

// cube_
template <class T, size_t M, size_t N, size_t L>
using cube_ =
    tensor<T,
           tensor_shape<size_t, const_size<M>, const_size<N>, const_size<L>>>;
using cube2 = cube_<double, 2, 2, 2>;
using cube3 = cube_<double, 3, 3, 3>;

// cubex_
template <class T>
using cubex_ = tensor<T, tensor_shape<size_t, size_t, size_t, size_t>>;
using cubex = matx_<double>;

// strings
using str = vecx_<char>;
using wstr = vecx_<wchar_t>;
using u16str = vecx_<char16_t>;
using u32str = vecx_<char32_t>;

// gstrings
template <class T> using gstr = gvecx_<char, T>;
template <class T> using gwstr = gvecx_<wchar_t, T>;
template <class T> using gu16str = gvecx_<char16_t, T>;
template <class T> using gu32str = gvecx_<char32_t, T>;

// tensor_of_rank
namespace details {
template <class T, class SeqT> struct _make_tensor_of_rank_seq {
  using type = void;
};
template <class T, size_t... Is>
struct _make_tensor_of_rank_seq<T, const_ints<size_t, Is...>> {
  using type = tensor<T, tensor_shape<size_t, always_t<size_t, Is>...>>;
};
}
template <class T, size_t Rank>
using tensor_of_rank = typename details::_make_tensor_of_rank_seq<
    T, decltype(make_const_sequence(const_size<Rank>()))>::type;

template <class ET = double, class ST, class... SizeTs, class RNG>
inline tensor<ET, tensor_shape<ST, SizeTs...>>
rand(const tensor_shape<ST, SizeTs...> &shape, RNG &rng);
}
