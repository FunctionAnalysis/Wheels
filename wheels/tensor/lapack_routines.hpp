#pragma once

#ifdef _MSC_VER
#include <complex.h>
#define lapack_complex_float _C_float_complex
#define lapack_complex_double _C_double_complex
#include <lapacke.h>
#ifdef I
#undef I
#endif
#else
#include <lapacke.h>
#endif

#include <complex>

namespace wheels {
namespace lapack {

// https://software.intel.com/en-us/node/521115
inline void gelsd(int m, int n, int nrhs, float *a, int lda, float *b,
                  int ldb, float *s, float *rcond, int *rank, float *work,
                  int *lwork, int *iwork, int *info) {
  sgelsd_(&m, &n, &nrhs, a, &lda, b, &ldb, s, rcond, rank, work, lwork, iwork, info);
}
inline void gelsd(int *m, int *n, int *nrhs, double *a, int *lda, double *b,
                  int *ldb, double *s, double *rcond, int *rank, double *work,
                  int *lwork, int *iwork, int *info) {
  dgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork, info);
}
inline void gelsd(int *m, int *n, int *nrhs, std::complex<float> *a, int *lda,
                  std::complex<float> *b, int *ldb, float *s, float *rcond,
                  int *rank, std::complex<float> *work, int *lwork,
                  float *rwork, int *iwork, int *info) {
  cgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, rwork, iwork,
          info);
}
inline void gelsd(int *m, int *n, int *nrhs, std::complex<double> *a, int *lda,
                  std::complex<double> *b, int *ldb, double *s, double *rcond,
                  int *rank, std::complex<double> *work, int *lwork,
                  double *rwork, int *iwork, int *info) {
  zgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, rwork, iwork,
          info);
}
}
}
