#pragma once

#include "../core/macros.hpp"

#if defined(wheels_compiler_msc)
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

#include "../core/types.hpp"
#include <complex>

#define wheels_lapack(fun) fun##_

namespace wheels {

using blas_int = int;

namespace lapack {

template <class TT>
inline void gels(char *trans, blas_int *m, blas_int *n, blas_int *nrhs, TT *a,
                 blas_int *lda, TT *b, blas_int *ldb, TT *work, blas_int *lwork,
                 blas_int *info) {

  if (types<TT>() == types<float>()) {
    typedef float T;
    wheels_lapack(sgels)(trans, m, n, nrhs, (T *)a, lda, (T *)b, ldb, (T *)work,
                         lwork, info);
  } else if (types<TT>() == types<double>()) {
    typedef double T;
    wheels_lapack(dgels)(trans, m, n, nrhs, (T *)a, lda, (T *)b, ldb, (T *)work,
                         lwork, info);
  } else if (types<TT>() == types<std::complex<float>>()) {
    typedef std::complex<float> T;
    wheels_lapack(cgels)(trans, m, n, nrhs, (T *)a, lda, (T *)b, ldb, (T *)work,
                         lwork, info);
  } else if (types<TT>() == types<std::complex<double>>()) {
    typedef std::complex<double> T;
    wheels_lapack(zgels)(trans, m, n, nrhs, (T *)a, lda, (T *)b, ldb, (T *)work,
                         lwork, info);
  }
}

template <class TT>
inline void getrf(blas_int *m, blas_int *n, TT *a, blas_int *lda,
                  blas_int *ipiv, blas_int *info) {
  if (types<TT>() == types<float>()) {
    typedef float T;
    wheels_lapack(sgetrf)(m, n, (T *)a, lda, ipiv, info);
  } else if (types<TT>() == types<double>()) {
    typedef double T;
    wheels_lapack(dgetrf)(m, n, (T *)a, lda, ipiv, info);
  } else if (types<TT>() == types<std::complex<float>>()) {
    typedef std::complex<float> T;
    wheels_lapack(cgetrf)(m, n, (T *)a, lda, ipiv, info);
  } else if (types<TT>() == types<std::complex<double>>()) {
    typedef std::complex<double> T;
    wheels_lapack(zgetrf)(m, n, (T *)a, lda, ipiv, info);
  }
}

template <class TT>
inline void getri(blas_int *n, TT *a, blas_int *lda, blas_int *ipiv, TT *work,
                  blas_int *lwork, blas_int *info) {
  if (types<TT>() == types<float>()) {
    typedef float T;
    wheels_lapack(sgetri)(n, (T *)a, lda, ipiv, (T *)work, lwork, info);
  } else if (types<TT>() == types<double>()) {
    typedef double T;
    wheels_lapack(dgetri)(n, (T *)a, lda, ipiv, (T *)work, lwork, info);
  }
  if (types<TT>() == types<std::complex<float>>()) {
    typedef std::complex<float> T;
    wheels_lapack(cgetri)(n, (T *)a, lda, ipiv, (T *)work, lwork, info);
  }
  if (types<TT>() == types<std::complex<double>>()) {
    typedef std::complex<double> T;
    wheels_lapack(zgetri)(n, (T *)a, lda, ipiv, (T *)work, lwork, info);
  }
}
}
}
