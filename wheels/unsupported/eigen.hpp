#pragma once

#include <Eigen/Dense>

#include "../tensor/map.hpp"

namespace wheels {
template <class Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
constexpr auto
map(const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> &m) {
  // todo
  // 
  return map(make_shape(conditional(const_bool<Rows == Eigen::Dynamic>(),
                                    m.rows(), const_int<Rows>()),
                        conditional(const_bool<Cols == Eigen::Dynamic>(),
                                    m.cols(), const_int<Cols>())),
             m.data());
}

}
