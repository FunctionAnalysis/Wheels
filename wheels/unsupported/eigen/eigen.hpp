#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>

#include "../../src/tensor_map.hpp"
#include "../../src/permute.hpp"

namespace wheels {
namespace details {
template <class RowT, class ColT, class T>
constexpr auto _map_eigen_matrix_conditionally(RowT rows, ColT cols, T *data,
                                               yes) {
  return map(make_shape(rows, cols), data);
}
template <class RowT, class ColT, class T>
constexpr auto _map_eigen_matrix_conditionally(RowT rows, ColT cols, T *data,
                                               no) {
  return transpose(map(make_shape(cols, rows), data));
}
}

// map
template <class Scalar, int Rows, int Cols, int Options>
constexpr auto
map(const Eigen::Matrix<Scalar, Rows, Cols, Options, Rows, Cols> &m) {
  return details::_map_eigen_matrix_conditionally(
      conditional(const_bool<Rows == Eigen::Dynamic>(), m.rows(),
                  const_int<Rows>()),
      conditional(const_bool<Cols == Eigen::Dynamic>(), m.cols(),
                  const_int<Cols>()),
      m.data(), const_bool<Options & Eigen::RowMajor>());
}

template <class Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
inline auto map(Eigen::Matrix<Scalar, Rows, Cols, Options, Rows, Cols> &m) {
  return details::_map_eigen_matrix_conditionally(
      conditional(const_bool<Rows == Eigen::Dynamic>(), m.rows(),
                  const_int<Rows>()),
      conditional(const_bool<Cols == Eigen::Dynamic>(), m.cols(),
                  const_int<Cols>()),
      m.data(), const_bool<Options & Eigen::RowMajor>());
}
}
