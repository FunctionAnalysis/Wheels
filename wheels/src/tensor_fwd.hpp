/* * *
 * The MIT License (MIT)
 * 
 * Copyright (c) 2016 Hao Yang (yangh2007@gmail.com)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * * */

#pragma once

#include "utility_fwd.hpp"

#include "shape_fwd.hpp"
#include "tensor_base_fwd.hpp"

namespace wheels {
template <class ET, class ShapeT> class tensor;

// vec_
template <class T, size_t N>
using vec_ = tensor<T, tensor_shape<size_t, const_size<N>>>;
using vec2 = vec_<double, 2>;
using vec3 = vec_<double, 3>;
using vec4 = vec_<double, 4>;

// gvec_
template <class E, size_t N, class T>
using gvec_ = tensor_base<E, tensor_shape<size_t, const_size<N>>, T>;
template <class T> using gvec2 = gvec_<double, 2, T>;
template <class T> using gvec3 = gvec_<double, 3, T>;
template <class T> using gvec4 = gvec_<double, 4, T>;

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
using mat4 = mat_<double, 4, 4>;

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
using cube4 = cube_<double, 4, 4, 4>;

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
namespace detail {
template <class T, class SeqT> struct _make_tensor_of_rank_seq {
  using type = void;
};
template <class T, size_t... Is>
struct _make_tensor_of_rank_seq<T, const_ints<size_t, Is...>> {
  using type = tensor<T, tensor_shape<size_t, always_t<size_t, Is>...>>;
};
}
template <class T, size_t Rank>
using tensor_of_rank = typename detail::_make_tensor_of_rank_seq<
    T, decltype(make_const_sequence(const_size<Rank>()))>::type;

template <class ET = double, class ST, class... SizeTs, class RNG>
inline tensor<ET, tensor_shape<ST, SizeTs...>>
rand(const tensor_shape<ST, SizeTs...> &shape, RNG &rng);
}
