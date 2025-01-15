////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "matx/core/sparse_tensor.h"

namespace matx {
namespace experimental {

//
// MatX uses a single versatile sparse tensor type that uses a tensor format
// DSL (Domain Specific Language) to describe a vast space of storage formats.
// This file provides a number of convenience factory methods that construct
// sparse matrices in well-known storage formats, like COO, CSR, and CSC,
// directly from the constituent buffers. More factory methods can easily be
// added as the need arises.
//

// Internal helper method for sparse matrices (2-dim tensors).
template <typename VAL, typename CRD, typename POS, typename TF>
static __MATX_INLINE__ auto
make_sparse_matrix(const index_t (&shape)[2], VAL *val, size_t vsz, CRD *c0,
                   size_t c0sz, CRD *c1, size_t c1sz, POS *p0, size_t p0sz,
                   POS *p1, size_t p1sz, bool owning) {
  // Values.
  raw_pointer_buffer<VAL, matx_allocator<VAL>> bufv{val, vsz, owning};
  basic_storage<decltype(bufv)> sv{std::move(bufv)};
  // Coordinates.
  raw_pointer_buffer<CRD, matx_allocator<CRD>> bufc0{c0, c0sz, owning};
  raw_pointer_buffer<CRD, matx_allocator<CRD>> bufc1{c1, c1sz, owning};
  basic_storage<decltype(bufc0)> sc0{std::move(bufc0)};
  basic_storage<decltype(bufc1)> sc1{std::move(bufc1)};
  // Positions.
  raw_pointer_buffer<POS, matx_allocator<POS>> bufp0{p0, p0sz, owning};
  raw_pointer_buffer<POS, matx_allocator<POS>> bufp1{p1, p1sz, owning};
  basic_storage<decltype(bufp0)> sp0{std::move(bufp0)};
  basic_storage<decltype(bufp1)> sp1{std::move(bufp1)};
  // Sparse tensor in templated format.
  return sparse_tensor_t<VAL, CRD, POS, TF, decltype(sv), decltype(sc0),
                         decltype(sp0)>(shape, std::move(sv),
                                        {std::move(sc0), std::move(sc1)},
                                        {std::move(sp0), std::move(sp1)});
}

// Constructs a sparse matrix in COO format directly from the values and
// coordinates arrays. The entries should be sorted by row, then column.
// Duplicate entries should not occur. Explicit zeros may be stored.
template <typename VAL, typename CRD>
auto make_coo(VAL *val, CRD *row, CRD *col, const index_t (&shape)[2],
              size_t nse, bool owning = false) {
  // No positions, although some forms use [0,nse] in first.
  const size_t vsz = nse * sizeof(VAL);
  const size_t csz = nse * sizeof(CRD);
  return make_sparse_matrix<VAL, CRD, int, COO>(
      shape, val, vsz, row, csz, col, csz, nullptr, 0, nullptr, 0, owning);
}

// As above for COO, operating directly on MatX 1-dim tensors for the buffers.
template <typename ValTensor, typename CrdTensor>
auto make_coo(ValTensor &val, CrdTensor &row, CrdTensor &col,
              const index_t (&shape)[2], bool owning = false) {
  const size_t nse = val.Size(0);
  return make_coo(val.Data(), row.Data(), col.Data(), shape, nse, owning);
}

// Constructs a sparse matrix in CSR format directly from the values, the row
// positions, and column coordinates arrays. The entries should be sorted by
// row, then column. Explicit zeros may be stored.
template <typename VAL, typename CRD, typename POS>
auto make_csr(VAL *val, POS *rowp, CRD *col, const index_t (&shape)[2],
              size_t nse, bool owning = false) {
  const size_t vsz = nse * sizeof(VAL);
  const size_t csz = nse * sizeof(CRD);
  const size_t psz = (shape[0] + 1) * sizeof(POS);
  return make_sparse_matrix<VAL, CRD, POS, CSR>(
      shape, val, vsz, nullptr, 0, col, csz, nullptr, 0, rowp, psz, owning);
}

// As above for CSR, operating directly on MatX 1-dim tensors for the buffers.
template <typename ValTensor, typename PosTensor, typename CrdTensor>
auto make_csr(ValTensor &val, PosTensor &rowp, CrdTensor &col,
              const index_t (&shape)[2], bool owning = false) {
  const size_t nse = val.Size(0);
  return make_csr(val.Data(), rowp.Data(), col.Data(), shape, nse, owning);
}

// Constructs a sparse matrix in CSC format directly from the values,
// the row coordinates, and column poc0tion arrays. The entries should
// be sorted by column, then row. Explicit zeros may be stored.
template <typename VAL, typename CRD, typename POS>
auto make_csc(VAL *val, CRD *row, POS *colp, const index_t (&shape)[2],
              size_t nse, bool owning = false) {
  const size_t vsz = nse * sizeof(VAL);
  const size_t csz = nse * sizeof(CRD);
  const size_t psz = (shape[1] + 1) * sizeof(POS);
  return make_sparse_matrix<VAL, CRD, POS, CSR>(
      shape, val, vsz, nullptr, 0, row, csz, nullptr, 0, colp, psz, owning);
}

// As above for CSC, operating directly on MatX 1-dim tensors for the buffers.
template <typename ValTensor, typename CrdTensor, typename PosTensor>
auto make_csc(ValTensor &val, CrdTensor &row, PosTensor &colp,
              const index_t (&shape)[2], bool owning = false) {
  const size_t nse = val.Size(0);
  return make_csc(val.Data(), row.Data(), colp.Data(), shape, nse, owning);
}

} // namespace experimental
} // namespace matx
