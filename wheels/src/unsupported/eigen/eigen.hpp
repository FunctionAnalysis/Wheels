#pragma once

#include <Eigen/Dense>

#include "../../tensor/map.hpp"
#include "../../tensor/permute.hpp"

namespace wheels {
namespace details {
template <class T> constexpr auto _transpose_or_not(T &&in, yes) {
  return transpose(forward<T>(in));
}
template <class T> constexpr T &&_transpose_or_not(T &&in, no) {
  return static_cast<T &&>(in);
}
}

// map
template <class Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
constexpr auto
map(const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> &m) {
  return details::_transpose_or_not(
      map(make_shape(conditional(const_bool<Rows == Eigen::Dynamic>(), m.rows(),
                                 const_int<Rows>()),
                     conditional(const_bool<Cols == Eigen::Dynamic>(), m.cols(),
                                 const_int<Cols>())),
          m.data()),
      const_bool<Options & Eigen::RowMajor>());
}

template <class Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
inline auto
map(Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> &m) {
  return details::_transpose_or_not(
      map(make_shape(conditional(const_bool<Rows == Eigen::Dynamic>(), m.rows(),
                                 const_int<Rows>()),
                     conditional(const_bool<Cols == Eigen::Dynamic>(), m.cols(),
                                 const_int<Cols>())),
          m.data()),
      const_bool<Options & Eigen::RowMajor>());
}
}
