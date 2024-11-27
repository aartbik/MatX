#pragma once

#include <string>

#include "matx/core/sparse_tensor_format.h"
#include "matx/core/storage.h"
#include "matx/core/tensor_impl.h"
#include "matx/core/tensor_utils.h"

namespace matx {

// Sparse tensors
//   VAL : data type of elements
//   CRD : data type of coordinates
//   POS : data type of positions
//   DIM : dimension rank (rank of user facing tensor)
//   LVL : level rank (rank of stored tensor)
template <typename VAL, typename CRD, typename POS, int DIM, int LVL,
          typename StorageV = DefaultStorage<VAL>,
          typename StorageC = DefaultStorage<CRD>,
          typename StorageP = DefaultStorage<POS>,
          typename DimDesc = DefaultDescriptor<DIM>>
class sparse_tensor_t : public detail::tensor_impl_t<VAL, DIM, DimDesc> {
public:
  //
  // Constructs a sparse tensor with given shape, format, and storage sizes.
  // The storage sizes denote the static capacity needed for the values and,
  // in level order, the coordinates and positions.
  //
  // Most users should not use this constructor directly. Instead use
  // the "make_sparse_tensor" methods.
  //
  // A sample constructor call is
  //
  //   sparse_tensor_t<float, int, int, 2, 2>
  //      As({m,n}, COO, {10,10,0,10,0});
  //
  // which constructs a m x n sparse matrix As in coordinate scheme format
  // (both dimension and level rank are 2) with 10 nonzero elements of type
  // float (the values) and 10 coordinates for dim-0 and dim-1 of type int
  // (in the crd[0] and crd[1] arrays). The positions are unused for COO.
  //
  __MATX_INLINE__
  sparse_tensor_t(const typename DimDesc::shape_type (&shape)[DIM],
                  const TensorFormat<DIM, LVL> &f,
                  const index_t (&sizes)[2 * LVL + 1])
      : detail::tensor_impl_t<VAL, DIM, DimDesc>(shape), format(f),
        values_{typename StorageV::container{sizes[0] * sizeof(VAL)}} {
    // Initialize coordinates and positions arrays with own containers.
    for (int l = 0, s = 1; l < LVL; l++) {
      const index_t csz = sizes[s++];
      if (csz)
        coordinates_[l] = typename StorageC::container{csz * sizeof(CRD)};
      const index_t psz = sizes[s++];
      if (psz)
        positions_[l] = typename StorageP::container{psz * sizeof(POS)};
    }
    // Superclass tensor_impl has DimDesc and values_ pointer
    //
    // TODO(cliff): what about the others?
    //
    this->SetLocalData(values_.data());
  }

  // Default destructor.
  __MATX_INLINE__ ~sparse_tensor_t() = default;

  // Identifying string.
  __MATX_INLINE__ const std::string str() const {
    return std::string("SpT") + std::to_string(DIM) + ":" +
           std::to_string(LVL) + "_" + detail::to_short_str<VAL>() + "_" +
           detail::to_short_str<CRD>() + "_" + detail::to_short_str<POS>();
  }

  // Number of stored elements.
  index_t getNse() const { return values_.size() / sizeof(VAL); }

private:
  // The tensor format describes how a sparse tensor is stored. It
  // provides an implicit mapping from the tensor dimensions to the
  // tensor levels (and the inverse back).
  const TensorFormat<DIM, LVL> format;

  // Primary storage of sparse tensor (explicitly stored element values).
  StorageV values_;

  // Secondary storage of sparse tensor (coordinates and positions).
  // There is potentially one for each level, although some of these
  // may remain empty. The secondary storage is essential to determine
  // where in the original tensor the explicitly stored elements reside.
  StorageC coordinates_[LVL];
  StorageP positions_[LVL];
};

} // end namespace matx
