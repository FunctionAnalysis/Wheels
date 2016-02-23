#pragma once

#include "../tensor.hpp"
#include "lapack.hpp"

namespace wheels {
namespace auxmath {

// solve min |AX - B|
// A: m x n
// B: m x nrhs
// return: n x nrhs
template <class ET, class ST1, class MT1, class NT1, class T1, class ST2,
          class MT2, class NT2, class T2>
auto solve(const tensor_base<ET, tensor_shape<ST1, MT1, NT1>, T1> &A,
           const tensor_base<ET, tensor_shape<ST2, MT2, NT2>, T2> &B,
           bool *succeed = nullptr) {
  char trans = 'N'; // 'T' if is transposed
  blas_int m = (blas_int)A.rows();
  blas_int n = (blas_int)A.cols();
  blas_int lda = max(1, m);
  // Adata: lda x n
  tensor<ET, tensor_shape<size_t, size_t, size_t>> Adata;
  if (lda == m) {
    Adata = A.t();
  } else {
    Adata = cat_at(const_index<0>(), A.derived(), zeros(make_shape(lda - m, n)))
                .t();
  }

  assert(m == B.rows());
  blas_int nrhs = (blas_int)B.cols();
  blas_int ldb = max(1, m, n);
  // Bdata: ldb x nrhs
  tensor<ET, tensor_shape<size_t, size_t, size_t>> Bdata;
  if (ldb == m) {
    Bdata = B.t();
  } else {
    Bdata =
        cat_at(const_index<0>(), B.derived(), zeros(make_shape(ldb - m, nrhs)))
            .t();
  }

  // work
  blas_int lwork = max(1, min(m, n) + max(min(m, n), nrhs) * 1);
  vecx_<ET> work(make_shape(lwork));

  blas_int info = 0;
  lapack::gels(&trans, &m, &n, &nrhs, Adata.ptr(), &lda, Bdata.ptr(), &ldb,
               work.ptr(), &lwork, &info);
  if (succeed) {
    *succeed = info == 0;
  }

  return std::move(Bdata).t().block(make_range(0, n), index_tags::everything);
}

// solve min |AX - B|
// A: m x n
// B: m vector
// return: n vector
template <class ET, class ST1, class MT1, class NT1, class T1, class ST2,
          class MT2, class T2>
auto solve(const tensor_base<ET, tensor_shape<ST1, MT1, NT1>, T1> &A,
           const tensor_base<ET, tensor_shape<ST2, MT2>, T2> &B,
           bool *succeed = nullptr) {
  char trans = 'N'; // 'T' if is transposed
  blas_int m = (blas_int)A.rows();
  blas_int n = (blas_int)A.cols();
  blas_int lda = max(1, m);

  // Adata: lda x n
  tensor<ET, tensor_shape<size_t, size_t, size_t>> Adata;
  if (lda == m) {
    Adata = A.t();
  } else {
    Adata = cat_at(const_index<0>(), A.derived(), zeros(make_shape(lda - m, n)))
                .t();
  }

  assert(m == (blas_int)B.numel());
  blas_int nrhs = 1;
  blas_int ldb = max(1, m, n);
  // Bdata: ldb
  tensor<ET, tensor_shape<size_t, size_t>> Bdata;
  if (ldb == m) {
    Bdata = B;
  } else {
    Bdata = cat_at(const_index<0>(), B.derived(), zeros(make_shape(ldb - m)));
  }

  // work
  blas_int lwork = max(1, min(m, n) + max(min(m, n), nrhs) * 2);
  vecx_<ET> work(make_shape(lwork));

  blas_int info = 0;
  lapack::gels(&trans, &m, &n, &nrhs, Adata.ptr(), &lda, Bdata.ptr(), &ldb,
               work.ptr(), &lwork, &info);
  if (succeed) {
    *succeed = info == 0;
  }

  return std::move(Bdata).block(make_range(0, n));
}
}
}
