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
inline void sgelsd(int *m, int *n, int *nrhs, float *a, int *lda, float *b,
                   int *ldb, float *s, float *rcond, int *rank, float *work,
                   int *lwork, int *iwork, int *info) {
  sgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork, info);
}
inline void dgelsd(int *m, int *n, int *nrhs, double *a, int *lda, double *b,
                   int *ldb, double *s, double *rcond, int *rank, double *work,
                   int *lwork, int *iwork, int *info) {
  dgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, iwork, info);
}
inline void cgelsd(int *m, int *n, int *nrhs, std::complex<float> *a, int *lda,
                   std::complex<float> *b, int *ldb, float *s, float *rcond,
                   int *rank, std::complex<float> *work, int *lwork,
                   float *rwork, int *iwork, int *info) {
  cgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, rwork, iwork,
          info);
}
inline void zgelsd(int *m, int *n, int *nrhs, std::complex<double> *a, int *lda,
                   std::complex<double> *b, int *ldb, double *s, double *rcond,
                   int *rank, std::complex<double> *work, int *lwork,
                   double *rwork, int *iwork, int *info) {
  zgelsd_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, rwork, iwork,
          info);
}
}

namespace lapack_tuned {
#define OUT
#define INOUT
inline int gelsd(int m, int n, int nrhs, INOUT float *a, int lda,
                 INOUT float *b, int ldb, OUT float *s, float rcond,
                 OUT int &rank) {
  int info = 0;
  lapack::sgelsd(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, nullptr,
                 nullptr, nullptr, &info);
  return info;
}
inline int gelsd(int m, int n, int nrhs, INOUT double *a, int lda,
                 INOUT double *b, int ldb, OUT double *s, double rcond,
                 OUT int &rank) {
  int info = 0;
  lapack::dgelsd(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, nullptr,
                 nullptr, nullptr, &info);
  return info;
}
inline int gelsd(int m, int n, int nrhs, INOUT std::complex<float> *a, int lda,
                 INOUT std::complex<float> *b, int ldb, OUT float *s,
                 float rcond, OUT int &rank) {
  int info = 0;
  lapack::cgelsd(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, nullptr,
                 nullptr, nullptr, nullptr, &info);
  return info;
}
inline int gelsd(int m, int n, int nrhs, INOUT std::complex<double> *a, int lda,
                 INOUT std::complex<double> *b, int ldb, OUT double *s,
                 double rcond, OUT int &rank) {
  int info = 0;
  lapack::zgelsd(&m, &n, &nrhs, a, &lda, b, &ldb, s, &rcond, &rank, nullptr,
                 nullptr, nullptr, nullptr, &info);
  return info;
}
#undef INOUT
#undef OUT
}
}
