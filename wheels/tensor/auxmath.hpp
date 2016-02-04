#pragma once

#if defined(wheels_with_opencv)
#include "../unsupported/opencv.hpp"
#endif

#if defined(wheels_with_eigen)
#include "../unsupported/eigen.hpp"
#endif

#include "tensor.hpp"

namespace wheels {
namespace auxmath {

// Solve a system of linear equations
template <class ET, class ShapeT1, class T1, class ShapeT2, class T2,
          class ShapeT3, class T3>
bool solve(tensor_continuous_data_base<ShapeT1, ET, T1> &out,
           tensor_continuous_data_base<ShapeT2, ET, T2> &A,
           const tensor_base<ShapeT3, ET, T3> &X) {
#if defined(wheels_with_opencv)
  { return false; }
#endif

#if defined(wheels_with_eigen)
  { 
    
  }
#endif
  return false;
}
}
}
