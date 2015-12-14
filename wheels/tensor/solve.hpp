#pragma once

#include <complex>

extern "C" {
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#define lapack_complex_float_real(z) (z.real())
#define lapack_complex_float_imag(z) (z.image())
#define lapack_complex_double_real(z) (z.real())
#define lapack_complex_double_imag(z) (z.imag())

lapack_complex_float lapack_make_complex_float(float re, float im) {
  return std::complex<float>(re, im);
}
lapack_complex_double lapack_make_complex_double(double re, double im) {
  return std::complex<double>(re, im);
}
}

#include <cblas.h>
#include <lapacke.h>

#include "base.hpp"

namespace wheels {

template <class ShapeT, class ET, class AT, class BT>
class solve_result
    : public tensor_base<ShapeT, ET, solve_result<ShapeT, ET, AT, BT>> {};

// min ||a*x-b||
template <class MT1, class NT1, class ET1, class T1, class MT2, class ET2,
          class T2, class MT3, class ET3, class T3>
auto solve(const tensor_base<tensor_shape<size_t, MT1, NT1>, ET1, T1> &a,
           const tensor_base<tensor_shape<size_t, MT2>, ET2, T2> &b) {
  assert(a.size(const_index<0>()) == b.size(const_index<0>()));
  throw std::runtime_error("not implemented yet");
}

template <class ST1, class ST2, class T1, class T2>
auto solve(const tensor_base<tensor_shape<ST1, ST1, ST1>, double, T1> &a,
           const tensor_base<tensor_shape<ST2, ST2, ST2>, double, T2> &b) {
  assert(a.size(const_index<0>()) == b.size(const_index<0>()));
  auto av = a.eval();
  auto bv = b.eval();
  // LAPACK_dgelsd(av.rows(), av.cols(), bv.cols(), av.data(), av.rows(),
  // bv.data(), )
  throw std::runtime_error("not implemented yet");
}
}