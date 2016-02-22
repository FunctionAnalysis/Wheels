#pragma once

#include "../tensor.hpp"
#include "lapack.hpp"

namespace wheels {
template <class T, size_t Depth> class cv_image;
namespace auxmath {

// solve min |AX - B|
// A: m x n
// B: m x nrhs
// X: n x nrhs
template <class ET, class ST1, class MT1, class NT1, class T1, class ST2,
          class MT2, class NT2, class T2, class ST3, class MT3, class NT3,
          class T3>
bool solve(const tensor_base<ET, tensor_shape<ST1, MT1, NT1>, T1> &A,
           const tensor_base<ET, tensor_shape<ST2, MT2, NT2>, T2> &B,
           tensor_base<ET, tensor_shape<ST3, MT3, NT3>, T3> &X) {
  char trans = 'N'; // 'T' if is transposed
  blas_int m = (blas_int)A.rows();
  blas_int n = (blas_int)A.cols();
  blas_int lda = m;
  tensor<ET, tensor_shape<ST1, MT1, NT1>> Adata = A;

  blas_int nrhs = (blas_int)B.cols();
  blas_int ldb = (blas_int)B.rows();
  tensor<ET, tensor_shape<ST2, MT2, NT2>> Bdata = B;

  blas_int lwork = max(1, min(m, n) + max(min(m, n), nrhs) * 1);
  vecx_<ET> work(make_shape(lwork));

  blas_int info = 0;
  lapack::gels(&trans, &m, &n, &nrhs, Adata.ptr(), &lda, Bdata.ptr(), &ldb,
               work.ptr(), &lwork, &info);

  if (m >= n) {
    X.derived() = Bdata.block(make_range(0, n), index_tags::everything);
  } else {
    X.derived() =
        cat_at(const_index<0>(), Bdata, zeros(make_shape(n - m, nrhs)));
  }

  return info == 0;
}

template <class ET, class ST1, class MT1, class NT1, class T1, class ST2,
          class MT2, class T2, class ST3, class MT3, class T3>
bool solve(const tensor_base<ET, tensor_shape<ST1, MT1, NT1>, T1> &A,
           const tensor_base<ET, tensor_shape<ST2, MT2>, T2> &B,
           tensor_base<ET, tensor_shape<ST3, MT3>, T3> &X) {
  char trans = 'N'; // 'T' if is transposed
  blas_int m = (blas_int)A.rows();
  blas_int n = (blas_int)A.cols();
  blas_int lda = m;
  tensor<ET, tensor_shape<ST1, MT1, NT1>> Adata = A;

  blas_int nrhs = 1;
  assert(m == (blas_int)B.numel());
  blas_int ldb = (blas_int)B.numel();
  tensor<ET, tensor_shape<ST2, MT2>> Bdata = B;

  blas_int lwork = max(1, min(m, n) + max(min(m, n), nrhs) * 2);
  vecx_<ET> work(make_shape(lwork));

  blas_int info = 0;
  lapack::gels(&trans, &m, &n, &nrhs, Adata.ptr(), &lda, Bdata.ptr(), &ldb,
               work.ptr(), &lwork, &info);

  if (m >= n) {
    X.derived() = Bdata.block(make_range(0, n));
  } else {
    X.derived() = cat(Bdata, zeros(make_shape(n - m)));
  }

  return info == 0;
}
}
}
