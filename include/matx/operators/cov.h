////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
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
// THIS SOFTWARE IS PROVIDED BY THE COpBRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COpBRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once


#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"
#include "matx/transforms/cov.h"

namespace matx
{
  namespace detail {
    template <typename OpA>
    class CovOp : public BaseOp<CovOp<OpA>>
    {
      private:
        typename detail::base_type_t<OpA> a_;
        cuda::std::array<index_t, OpA::Rank()> out_dims_;
        mutable detail::tensor_impl_t<typename remove_cvref_t<OpA>::value_type, OpA::Rank()> tmp_out_;
        mutable typename remove_cvref_t<OpA>::value_type *ptr = nullptr; 

      public:
        using matxop = bool;
        using value_type = typename OpA::value_type;
        using matx_transform_op = bool;
        using cov_xform_op = bool;

        __MATX_INLINE__ std::string str() const { 
          return "cov(" + get_type_str(a_) + ")";
        }

        __MATX_INLINE__ CovOp(const OpA &A) : 
              a_(A) {
          
          for (int r = 0; r < Rank(); r++) {
            out_dims_[r] = a_.Size(r);
          }
        }

        __MATX_HOST__ __MATX_INLINE__ auto Data() const noexcept { return ptr; }

        template <ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return tmp_out_.template operator()<EPT>(indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<detail::ElementsPerThread::ONE>(indices...);
        }

        template <OperatorCapability Cap>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(a_));
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return OpA::Rank();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return out_dims_[dim];
        }

        template <typename Out, typename Executor>
        void Exec(Out &&out, Executor &&ex) const {
          static_assert(is_cuda_executor_v<Executor>, "cov() only supports the CUDA executor currently");
          cov_impl(cuda::std::get<0>(out), a_, ex);
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }         
        }      

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
        {
          InnerPreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));      

          detail::AllocateTempTensor(tmp_out_, std::forward<Executor>(ex), out_dims_, &ptr);

          Exec(cuda::std::make_tuple(tmp_out_), std::forward<Executor>(ex));
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<OpA>()) {
            a_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }

          matxFree(ptr);
        }          
    };
  }


/**
 * Compute a covariance matrix without a plan
 *
 * Creates a new cov plan in the cache if none exists, and uses that to execute
 * the covariance calculation. This function is preferred over creating a plan
 * directly for both efficiency and simpler code. Since it only uses the
 * signature of the covariance to decide if a plan is cached, it may be able to
 * reused plans for different A matrices
 *
 * @tparam AType
 *    Data type of A operator
 *
 * @param a
 *   Covariance operator input view
 */
  template <typename AType>
    __MATX_INLINE__ auto cov(const AType &a)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
    
    return detail::CovOp(a);
  }

}
