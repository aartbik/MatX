#pragma once

#include <cusparse.h>

#include <cstdio>
#include <numeric>

#include "matx/core/cache.h"
#include "matx/core/sparse_tensor.h"
#include "matx/core/tensor.h"
#include "matx/transforms/matmul/matmul_common.h"

namespace matx {

namespace detail {

/**
 * Parameters needed to execute a CUSPARSE GEMM.
 */
struct MatMulCUSPARSEParams_t {
  MatXDataType_t dtype;
  int rank;
  cudaStream_t stream;
  float alpha;
  float beta;
  index_t nse;
  index_t m;
  index_t n;
  index_t k;
  int32_t batch;   // supported?
  index_t astride; // batch stride
  index_t bstride; // batch stride
  index_t cstride; // batch stride
  cusparseOperation_t opA;
  cusparseOperation_t opB;
  //
  // TODO(cliff): since we are sharing handles for A,B,C
  //       we need to make sure that this is captured
  //       in the params caching!!!!!!!
  //       is hash of Data() safe? or something else
  //
  // NOTE: this is different from cuBLAS where same plan
  //       can be shared with different data buffers!!!!
  //
};

template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB>
class MatMulCUSPARSEHandle_t {
public:
  using TA = typename TensorTypeA::value_type;
  using TB = typename TensorTypeB::value_type;
  using TC = typename TensorTypeC::value_type;
  static constexpr int RANKA = TensorTypeC::Rank();
  static constexpr int RANKB = TensorTypeC::Rank();
  static constexpr int RANKC = TensorTypeC::Rank();

  /**
   * Construct a sparse GEMM handle
   *   SpMV
   *   SpMM        <- for now
   *   SpGEMM
   *
   */
  MatMulCUSPARSEHandle_t(TensorTypeC &c, const TensorTypeA &a,
                         const TensorTypeB &b, cudaStream_t stream, float alpha,
                         float beta) {
    printf("create cusparse\n");
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

    static_assert(RANKA >= 2);
    static_assert(RANKB >= 2);
    static_assert(RANKC >= 2);
    MATX_ASSERT(a.Size(RANKA - 1) == b.Size(RANKB - 2), matxInvalidSize);
    MATX_ASSERT(c.Size(RANKC - 1) == b.Size(RANKB - 1), matxInvalidSize);
    MATX_ASSERT(c.Size(RANKC - 2) == a.Size(RANKA - 2), matxInvalidSize);

    params_ = GetGemmParams(c, a, b, stream, alpha, beta);

    ret = cusparseCreate(&handle_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);

    std::cout << "DATA " << a.str() << "," << b.str() << "," << c.str()
              << std::endl;

    void *crd_i = a.IData();
    void *crd_j = a.JData();
    void *val_a = a.Data();
    void *val_b = b.Data();
    void *val_c = c.Data();

    // Assumes COO and int coordinates.
    const cusparseIndexType_t idx = CUSPARSE_INDEX_32I;
    const cusparseIndexBase_t zb = CUSPARSE_INDEX_BASE_ZERO;
    ret = cusparseCreateCoo(&matA_, params_.m, params_.k, params_.nse, crd_i,
                            crd_j, val_a, idx, zb, MatXTypeToCudaType<TA>());
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);

    // Assumes two dense matrices.
    const cusparseOrder_t bo = CUSPARSE_ORDER_ROW;
    ret = cusparseCreateDnMat(&matB_, params_.k, params_.n, /*ld=*/params_.n,
                              val_b, MatXTypeToCudaType<TB>(), bo);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);
    const cusparseOrder_t bc = CUSPARSE_ORDER_ROW;
    ret = cusparseCreateDnMat(&matC_, params_.m, params_.n, /*ld=*/params_.n,
                              val_c, MatXTypeToCudaType<TC>(), bc);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);

    // Workspace.
    const cusparseSpMMAlg_t algo = CUSPARSE_SPMM_ALG_DEFAULT;
    const cudaDataType dtp =
        MatXTypeToCudaType<TC>(); // TODO: support separate computational type?!
    ret = cusparseSpMM_bufferSize(handle_, params_.opA, params_.opB,
                                  &params_.alpha, matA_, matB_, &params_.beta,
                                  matC_, dtp, algo, &workspaceSize_);
    MATX_ASSERT(ret == CUSPARSE_STATUS_SUCCESS, matxMatMulError);
    if (workspaceSize_)
      matxAlloc((void **)&workspace_, workspaceSize_, MATX_DEVICE_MEMORY);
  }

  ~MatMulCUSPARSEHandle_t() {
    printf("destroy cusparse\n");
    if (workspaceSize_)
      matxFree(workspace_);
    cusparseDestroy(handle_); // TODO: share handle between all cusparse ops?!
  }

  static detail::MatMulCUSPARSEParams_t
  GetGemmParams(TensorTypeC &c, const TensorTypeA &a, const TensorTypeB &b,
                cudaStream_t stream, float alpha, float beta) {
    detail::MatMulCUSPARSEParams_t params;
    params.dtype = TypeToInt<TC>();
    params.rank = c.Rank();
    params.stream = stream;
    params.alpha = alpha;
    params.beta = beta;

    // Batches
    params.batch = 1;
    params.astride = 0;
    params.bstride = 0;
    params.cstride = 0;

    // TODO: simple no-batch, row-wise, no-transpose for now, but fix this
    params.nse = 4;
    params.m = a.Size(TensorTypeA::Rank() - 2);
    params.n = b.Size(TensorTypeB::Rank() - 1);
    params.k = a.Size(TensorTypeB::Rank() - 1);
    params.opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    params.opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

    // TODO: make sure A,B,C are exactly accounted for!!!!

    return params;
  }

  __MATX_INLINE__ void Exec(TensorTypeC &c, const TensorTypeA &a,
                            const TensorTypeB &b) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL);
    MatMulDispatchA(a, b, c);
  }

private:
  cusparseHandle_t handle_ = nullptr;
  cusparseSpMatDescr_t matA_ = nullptr;
  cusparseDnMatDescr_t matB_ = nullptr;
  cusparseDnMatDescr_t matC_ = nullptr;
  cusparseStatus_t ret = CUSPARSE_STATUS_SUCCESS;
  size_t workspaceSize_ = 0;
  void *workspace_ = nullptr;
  detail::MatMulCUSPARSEParams_t params_;

  // Launch Matmul.
  __MATX_INLINE__ void MatMulLaunch(const TensorTypeA &a, const TensorTypeB &b,
                                    TensorTypeC &c) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    std::cout << "LAUNCH CUSPARSE: " << a.str() << "," << b.str() << ","
              << c.str() << std::endl;
    const cusparseSpMMAlg_t algo = CUSPARSE_SPMM_ALG_DEFAULT;
    const cudaDataType dtp =
        MatXTypeToCudaType<TC>(); // TODO: support separate computational type?!
    ret = cusparseSpMM(handle_, params_.opA, params_.opB, &params_.alpha, matA_,
                       matB_, &params_.beta, matC_, dtp, algo, workspace_);
  }

  // Dispatch C.
  inline void MatMulDispatchC(const TensorTypeA &a, const TensorTypeB &b,
                              TensorTypeC &c) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    MatMulLaunch(a, b, c);
  }

  // Dispatch B.
  inline void MatMulDispatchB(const TensorTypeA &a, const TensorTypeB &b,
                              TensorTypeC &c) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    MatMulDispatchC(a, b, c);
  }

  // Dispatch A.
  inline void MatMulDispatchA(const TensorTypeA &a, const TensorTypeB &b,
                              TensorTypeC &c) {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)
    MatMulDispatchB(a, b, c);
  }
};

/**
 * Crude hash on GEMM to get a reasonably good delta for collisions. This
 * doesn't need to be perfect, but fast enough to not slow down lookups, and
 * different enough so the common GEMM parameters change.
 */
struct MatMulCUSPARSEParamsKeyHash {
  std::size_t operator()(const MatMulCUSPARSEParams_t &k) const noexcept {
    return std::hash<uint64_t>()(k.nse) + std::hash<uint64_t>()(k.m) +
           std::hash<uint64_t>()(k.n) + std::hash<uint64_t>()(k.k) +
           std::hash<uint64_t>()(k.batch) +
           std::hash<uint64_t>()((size_t)k.stream);
  }
};

/**
 * Test GEMM parameters for equality. Unlike the hash, all parameters must
 * match.
 */
struct MatMulCUSPARSEParamsKeyEq {
  bool operator()(const MatMulCUSPARSEParams_t &l,
                  const MatMulCUSPARSEParams_t &t) const noexcept {
    return l.dtype == t.dtype && l.rank == t.rank && l.stream == t.stream &&
           l.alpha == t.alpha && l.beta == t.beta && l.nse == t.nse &&
           l.m == t.m && l.n == t.n && l.k == t.k && l.batch == t.batch &&
           l.astride == t.astride && l.bstride == t.bstride &&
           l.cstride == t.cstride && l.opA == t.opA && l.opB == t.opB;
  }
};

using gemm_cusparse_cache_t =
    std::unordered_map<MatMulCUSPARSEParams_t, std::any,
                       MatMulCUSPARSEParamsKeyHash, MatMulCUSPARSEParamsKeyEq>;

} // end namespace detail

template <typename Op>
__MATX_INLINE__ auto getCUSPARSESupportedTensor(const Op &in,
                                                cudaStream_t stream) {
  const auto support_func = [&in]() {
    if constexpr (is_tensor_view_v<Op>) {
      std::cout << "VIEW" << std::endl;
      return !((in.Stride(Op::Rank() - 1) != (index_t)1 &&
                in.Stride(Op::Rank() - 2) != (index_t)1) ||
               (in.Stride(Op::Rank() - 1) == (index_t)0 &&
                in.Size(Op::Rank() - 1) != (index_t)1) ||
               (in.Stride(Op::Rank() - 2) == (index_t)0 &&
                in.Size(Op::Rank() - 2) != (index_t)1));
    } else {
      std::cout << "NOVIEW" << std::endl;
      return true;
    }
  };
  std::cout << "MESS WITH " << in.str() << std::endl;
  return GetSupportedTensor(in, support_func, MATX_ASYNC_DEVICE_MEMORY, stream);
}

template <typename TensorTypeC, typename TensorTypeA, typename TensorTypeB>
void sparse_matmul_impl(TensorTypeC C, const TensorTypeA A, const TensorTypeB B,
                        const cudaExecutor &exec, float alpha = 1.0,
                        float beta = 0.0) {
  printf("**** cusparse matmul_impl (main entry)\n");
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  const auto stream = exec.getStream();

  // TODO: lots of simplifying assumptions for now

  auto c = getCUSPARSESupportedTensor(C, stream);
  auto a = A; // TODO: guaranteed sparse tensor due to current calling logic
  auto b = getCUSPARSESupportedTensor(B, stream);

  typedef decltype(c) ctype;
  typedef decltype(a) atype;
  typedef decltype(b) btype;

  // Get parameters required by these tensors (for caching).
  auto params =
      detail::MatMulCUSPARSEHandle_t<ctype, atype, btype>::GetGemmParams(
          c, a, b, stream, alpha, beta);

  // Lookup and cache.
  using cache_val_type = detail::MatMulCUSPARSEHandle_t<ctype, atype, btype>;
  detail::GetCache().LookupAndExec<detail::gemm_cusparse_cache_t>(
      detail::GetCacheIdFromType<detail::gemm_cusparse_cache_t>(), params,
      [&]() {
        return std::make_shared<cache_val_type>(c, a, b, stream, alpha, beta);
      },
      [&](std::shared_ptr<cache_val_type> cache_type) {
        cache_type->Exec(c, a, b);
      });
}

} // end namespace matx
