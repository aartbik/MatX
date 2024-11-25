#pragma once

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <numeric>
#include <type_traits>

#include "matx/core/allocator.h"
#include "matx/core/dlpack.h"
#include "matx/core/error.h"
#include "matx/core/sparse_tensor_format.h"
#include "matx/core/storage.h"
#include "matx/core/tensor_impl.h"
#include "matx/core/tensor_utils.h"
#include "matx/core/tie.h"
#include "matx/kernels/utility.cuh"

// forward declare
namespace matx {
template <typename T, int RANK, typename Storage, typename Desc> class tensor_t;
} // namespace matx

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
  // TODO: throw-away code just for demo,
  //       provide make_sparse_tensor()s instead
  __MATX_HOST__ sparse_tensor_t(tensor_t<VAL, DIM> &rhs) noexcept
      : detail::tensor_impl_t<VAL, DIM, DimDesc>(rhs.Shape()), format(COO),
        nse_(0), values_{typename StorageV::container{sizeof(VAL)}} {
    // Initialize coordinates and positions arrays containers.
    for (int l = 0; l < LVL; l++) {
      coordinates_[l] = typename StorageC::container{sizeof(CRD)};
      positions_[l] = typename StorageP::container{sizeof(POS)};
    }
    // Count NSE.
    for (size_t i = 0, ie = rhs.Size(0); i < ie; i++) {
      for (size_t j = 0, je = rhs.Size(1); j < je; j++) {
        if (rhs(i, j) != 0)
          nse_++;
      }
    }
    // TODO: Demo COO (but will be much more general!).
    coordinates_[0].allocate(nse_);
    coordinates_[1].allocate(nse_);
    values_.allocate(nse_);
    CRD *ibuf = coordinates_[0].data();
    CRD *jbuf = coordinates_[1].data();
    VAL *vbuf = values_.data();
    size_t k = 0;
    for (size_t i = 0, ie = rhs.Size(0); i < ie; i++) {
      for (size_t j = 0, je = rhs.Size(1); j < je; j++) {
        if (rhs(i, j) != 0) {
          // TODO: is this safe? are buffers on host still?
          ibuf[k] = static_cast<CRD>(i);
          jbuf[k] = static_cast<CRD>(j);
          vbuf[k] = rhs(i, j);
          k++;
        }
      }
    }
    assert(k == nse_);
    // Superclass tensor_impl has DimDesc and values_ pointer
    // TODO: what about the others?
    this->SetLocalData(values_.data());
  }

  __MATX_INLINE__ ~sparse_tensor_t() = default;

  __MATX_INLINE__ const std::string str() const {
    return std::string("SpT") + std::to_string(LVL) + ":" +
           std::to_string(LVL) + "_" + detail::to_short_str<VAL>() + "_" +
           detail::to_short_str<CRD>() + "_" + detail::to_short_str<POS>();
  }

  // For debugging.
  void print(size_t num = 16) {
    // TODO: I am not sure yet if we need LvlDesc stored on device
    //       too; for now I am just mapping DimDesc to dims/lvls.
    // Prepare dim and lvl sizes.
    size_t dims[DIM];
    size_t lvls[LVL];
    for (int d = 0; d < DIM; d++) {
      dims[d] = this->Size(d);
    }
    format.dim2lvl(dims, lvls, /*asSize=*/true);
    // Dump explicit and implicit contents.
    size_t sz = 1;
    size_t bytes = nse_ * sizeof(VAL);
    std::cout << "---- Sparse Tensor<" << detail::to_short_str<VAL>() << ","
              << detail::to_short_str<CRD>() << ","
              << detail::to_short_str<POS>() << ">" << std::endl;
    std::cout << "nse      : " << nse_ << std::endl;
    std::cout << "dim      : ";
    for (int d = 0; d < DIM; d++) {
      sz *= this->Size(d);
      std::cout << " " << dims[d];
      if (d != LVL - 1)
        std::cout << " x";
    }
    std::cout << std::endl;
    std::cout << "lvl      : ";
    for (int l = 0; l < LVL; l++) {
      std::cout << " " << lvls[l];
      if (l != LVL - 1)
        std::cout << " x";
    }
    std::cout << std::endl;
    std::cout << "format   : ";
    format.print();
    for (int r = 0; r < LVL; r++) {
      if (size_t e = 0) { // TODO: how?
        bytes += e * sizeof(POS);
        std::cout << "pos[" << r << "]   : (";
        for (size_t i = 0; i < e; i++) {
          if (i > num) {
            std::cout << " ...";
            break;
          }
          std::cout << " " << positions_[r].data()[i];
        }
        std::cout << " ) #" << e << std::endl;
      }
      if (size_t e = nse_) { // TODO: how?
        bytes += e * sizeof(CRD);
        std::cout << "crd[" << r << "]   : (";
        for (size_t i = 0; i < e; i++) {
          if (i > num) {
            std::cout << " ...";
            break;
          }
          std::cout << " " << coordinates_[r].data()[i];
        }
        std::cout << " ) #" << e << std::endl;
      }
    }
    std::cout << "values   : (";
    for (size_t i = 0; i < nse_; i++) {
      if (i > num) {
        std::cout << " ...";
        break;
      }
      std::cout << " " << (values_.data()[i]);
    }
    std::cout << " ) #" << nse_ << std::endl;
    std::cout << "data     : " << bytes << " bytes" << std::endl;
    const double sparsity =
        100.0 - (100.0 * static_cast<double>(nse_)) / static_cast<double>(sz);
    std::cout << "sparsity : " << sparsity << "%" << std::endl;
    std::cout << "----" << std::endl;
  }

  // TODO: replace with type trait logic
  bool is_sparse() const { return true; }

private:
  // Metadata the describes how sparse tensor is stored. It provides
  // an implicit mapping from the dimensions to the levels (and back).
  const TensorFormat<DIM, LVL> format;

  // Number of expliclitly stored elements (essentially size of values_).
  size_t nse_;

  // Primary storage of sparse tensor (explicitly stored element values).
  StorageV values_;

  // Secondary storage of sparse tensor (positions and coordinates).
  // There is potentially one for each level, although some of these
  // may remain empty. The secondary storage is essential to determine
  // where in the original tensor the explicitly stored elements reside.
  StorageP positions_[LVL];
  StorageC coordinates_[LVL];
};

} // end namespace matx
