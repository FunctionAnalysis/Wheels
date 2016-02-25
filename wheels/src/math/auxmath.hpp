#pragma once

#include "../tensor/block.hpp"
#include "../tensor/cat.hpp"
#include "../tensor/constants.hpp"
#include "../tensor/permute.hpp"
#include "../tensor/tensor.hpp"

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
  assert(m > 0 && n > 0);
  blas_int lda = m;
  // Adata: lda x n
  auto Adata = A.t().eval();

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

  return std::move(Bdata).t().block(make_interval(0, n),
                                    index_tags::everything);
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
  assert(m > 0 && n > 0);
  blas_int lda = m;

  // Adata: lda x n
  auto Adata = A.t().eval();

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
  blas_int lwork = max(1, min(m, n) + max(min(m, n), nrhs));
  vecx_<ET> work(make_shape(lwork));

  blas_int info = 0;
  lapack::gels(&trans, &m, &n, &nrhs, Adata.ptr(), &lda, Bdata.ptr(), &ldb,
               work.ptr(), &lwork, &info);
  if (succeed) {
    *succeed = info == 0;
  }

  return std::move(Bdata).block(make_interval(0, n));
}

// inverse n x n matrix
template <class ET, class ST, class MT, class NT, class T>
auto inverse(const tensor_base<ET, tensor_shape<ST, MT, NT>, T> &A,
             bool *succeed = nullptr) {
  assert(A.cols() == A.rows());
  blas_int n = (blas_int)A.rows();
  assert(n > 0);
  blas_int lda = n;
  auto Adata = A.t().eval();

  std::vector<blas_int> ipiv(n);

  blas_int lwork = n;
  vecx_<ET> work(make_shape(lwork));

  blas_int info = 0;
  // calling to getri causes heap corruption, FIXME!!
  lapack::getri(&n, Adata.ptr(), &lda, ipiv.data(), work.ptr(), &lwork, &info);
  if (succeed) {
    *succeed = info == 0;
  }

  return std::move(Adata).t();
}
}
}
