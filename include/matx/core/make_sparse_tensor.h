

#pragma once

#include "matx/core/sparse_tensor.h"
#include "matx/core/sparse_tensor_format.h"
#include "matx/core/storage.h"
#include "matx/core/tensor_desc.h"

namespace matx {

// General contruct sparse tensor from dense tensor.
template <typename VAL, typename CRD, typename POS, int DIM, int LVL,
          typename Storage, typename Desc>
sparse_tensor_t<VAL, CRD, POS, DIM, LVL>
make_sparse_tensor(tensor_t<VAL, DIM, Storage, Desc> const &rhs,
                   const TensorFormat<DIM, LVL> &format) {
  MATX_THROW(matxInvalidParameter, "tensor format not implemented yet");
}

// Specialized for matrices.
template <typename VAL, typename CRD, typename POS, typename Storage, typename Desc>
sparse_tensor_t<VAL, CRD, POS, 2, 2>
make_sparse_tensor(tensor_t<VAL, 2, Storage, Desc> const &rhs,
                   const TensorFormat<2, 2> &format) {
  // Handle COO.
  if (format.isCOO()) {
    const index_t m = rhs.Size(0), n = rhs.Size(1);
    index_t nse = 0;
    for (index_t i = 0; i < m; i++) {
      for (index_t j = 0; j < n; j++) {
        if (rhs(i, j) != 0)
          nse++;
      }
    }
    // TODO: fill the buffers of the sparse tensor
    return sparse_tensor_t<VAL, CRD, POS, 2, 2>({m,n}, format, {nse, nse, 0, nse, 0});
  }
  MATX_THROW(matxInvalidParameter, "tensor format not implemented yet");
}

} // namespace matx
