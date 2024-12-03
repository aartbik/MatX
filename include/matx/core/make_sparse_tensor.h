

#pragma once

#include "matx/core/sparse_tensor.h"
#include "matx/core/sparse_tensor_format.h"
#include "matx/core/storage.h"
#include "matx/core/tensor_desc.h"

namespace matx {

// General contruct sparse tensor from dense tensor.
template <typename VAL, typename CRD, typename POS, int DIM, int LVL>
sparse_tensor_t<VAL, CRD, POS, DIM, LVL>
make_sparse_tensor(tensor_t<VAL, DIM> const &rhs,
                   const TensorFormat<DIM, LVL> &format) {
  MATX_THROW(matxInvalidParameter, "tensor format not implemented yet");
}

// Specialized for matrices.
template <typename VAL, typename CRD, typename POS>
sparse_tensor_t<VAL, CRD, POS, 2, 2>
make_sparse_tensor(tensor_t<VAL, 2> const &rhs,
                   const TensorFormat<2, 2> &format) {
  // Handle COO.
  // 
  // TODO(cliff): templating on COO would be much better!
  //
  if (format.isCOO()) {
    const index_t m = rhs.Size(0), n = rhs.Size(1);
    index_t nse = 0;
    for (index_t i = 0; i < m; i++) {
      for (index_t j = 0; j < n; j++) {
        if (rhs(i, j) != 0)
          nse++;
      }
    }
    auto t = sparse_tensor_t<VAL, CRD, POS, 2, 2>({m, n}, format,
                                                  {nse, nse, 0, nse, 0});
    //
    // TODO(cliff): how to fill the buffers of the sparse tensor properly?
    //
    for (index_t i = 0, k = 0; i < m; i++) {
      for (index_t j = 0; j < n; j++) {
        if (rhs(i, j) != 0)
          t.setHack(k++, (CRD)i, (CRD)j, rhs(i, j));
      }
    }
    return t;
  }
  MATX_THROW(matxInvalidParameter, "tensor format not implemented yet");
}

} // namespace matx
