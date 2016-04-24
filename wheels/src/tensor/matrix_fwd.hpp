#pragma once

#include "base_fwd.hpp"

namespace wheels {

// matrix base
template <class T> struct matrix_base;

// auto matrix_mul(ts1, ts2);
template <class EleT, class ShapeT, class A, class B, bool AIsMat, bool BIsMat>
class matrix_mul_result;
}