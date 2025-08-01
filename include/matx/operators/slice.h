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


#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"

namespace matx
{
  /**
   * Slices elements from an operator/tensor.
   */
  namespace detail {

    template <int DIM, typename T, typename StrideType>
      class SliceOp : public BaseOp<SliceOp<DIM, T, StrideType>>
    {
      public:
        using value_type = typename T::value_type;
        using self_type = SliceOp<DIM, T, StrideType>;

      private:
        using shape_type = index_t;
        typename detail::base_type_t<T> op_;
        cuda::std::array<shape_type, DIM> sizes_;
        cuda::std::array<int32_t, DIM> dims_;
        cuda::std::array<shape_type, T::Rank()> starts_;
        StrideType strides_; // Add [[no_unique_address]] in c++20

      public:
        using matxop = bool;
        using matxoplvalue = bool;

        static_assert(T::Rank()>0, "SliceOp: Rank of operator must be greater than 0.");
        static_assert(DIM<=T::Rank(), "SliceOp: DIM must be less than or equal to operator rank.");

        __MATX_INLINE__ std::string str() const { return "slice(" + op_.str() + ")"; }

        __MATX_INLINE__ SliceOp(const T &op, const cuda::std::array<shape_type, T::Rank()> &starts,
                                      const cuda::std::array<shape_type, T::Rank()> &ends,
                                      StrideType strides) : op_(op) {
          int32_t d = 0;
          for(int32_t i = 0; i < T::Rank(); i++) {
            shape_type start = starts[i] < 0 ? op.Size(i) + starts[i] : starts[i];
            shape_type end   = ends[i]   < 0 ? op.Size(i) + ends[i]   : ends[i];

            MATX_ASSERT_STR((start > matxIdxSentinel) || (start < op.Size(i)), matxInvalidDim,
              "Slice slice index out of range of operator");
            MATX_ASSERT_STR((end > matxIdxSentinel) || (end <= op.Size(i)), matxInvalidDim,
              "Slice end index out of range of operator");

            starts_[i] = start;

            if constexpr (!std::is_same_v<NoStride, StrideType>) {
              strides_[i] = strides[i];
            }

            // compute dims and sizes
            if(end != matxDropDim) {
              MATX_ASSERT_STR(end != matxKeepDim, matxInvalidParameter, "matxKeepDim only valid for clone(), not slice()");

              dims_[d] = i;

              if(end == matxEnd) {
                sizes_[d] = op.Size(i) - start;
              } else {
                sizes_[d] = end - start;
              }

              //adjust size by stride
              if constexpr (!std::is_same_v<NoStride, StrideType>) {
                sizes_[d] = (shape_type)std::ceil(static_cast<double>(sizes_[d])/ static_cast<double>(strides_[d]));
              }

              d++;
            }
          }
          MATX_ASSERT_STR(d==Rank(), matxInvalidDim, "SliceOp: Number of dimensions without matxDropDim must equal new rank.");
        };

        template <detail::ElementsPerThread EPT, typename Op, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(
            Op&& op,
            const decltype(starts_) &starts,
            const decltype(strides_) &strides,
            const decltype(dims_) &dims,
            Is... indices)
        {
          if constexpr (EPT == ElementsPerThread::ONE) {
            static_assert(sizeof...(Is)==Rank());
            static_assert((std::is_convertible_v<Is, index_t> && ... ));

            // convert variadic type to tuple so we can read/update
            cuda::std::array<index_t, T::Rank()> ind = starts;
            cuda::std::array<index_t, Rank()> inds{indices...};

            MATX_LOOP_UNROLL
            for (int32_t i = 0; i < T::Rank(); i++) {
              MATX_LOOP_UNROLL
              for(int32_t j = 0; j < Rank(); j++) {
                if(dims[j] == i) {
                  if constexpr (!std::is_same_v<NoStride, StrideType>) {
                    ind[i] = starts[j] + inds[j] * strides[i];
                  }
                  else {
                    ind[i] = starts[j] + inds[j];
                  }
                }
              }
            }

            return get_value<EPT>(cuda::std::forward<Op>(op), ind);
          } else {
            return Vector<value_type, static_cast<index_t>(EPT)>{};
          }
        }

        template <detail::OperatorCapability Cap>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
          if constexpr (Cap == detail::OperatorCapability::ELEMENTS_PER_THREAD) {
            return detail::ElementsPerThread::ONE;
          }
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_));
        }

        template <detail::ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return get_impl<EPT>(cuda::std::as_const(op_), starts_, strides_, dims_, indices...);
        }

        template <detail::ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return get_impl<EPT>(cuda::std::forward<decltype(op_)>(op_), starts_, strides_, dims_, indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<detail::ElementsPerThread::ONE>(indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return this->operator()<detail::ElementsPerThread::ONE>(indices...);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return DIM;
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ shape_type Size(int32_t dim) const
        {
          return sizes_[dim];
        }

        ~SliceOp() = default;
        SliceOp(const SliceOp &rhs) = default;
        __MATX_INLINE__ auto operator=(const self_type &rhs) {
          return set(*this, rhs);
        }

        template<typename R>
        __MATX_INLINE__ auto operator=(const R &rhs) {
          if constexpr (is_matx_transform_op<R>()) {
            return mtie(*this, rhs);
          }
          else {
            return set(*this, rhs);
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }
    };
  }

#ifndef DOXYGEN_ONLY
  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue.
   *
   * @tparam OpType Input operator/tensor type
   * @param op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @param strides Optional:  the stride between consecutive elements
   * @return sliced operator
   */
  template <typename OpType>
  __MATX_INLINE__ auto slice( const OpType &op,
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends,
      const cuda::std::array<index_t, OpType::Rank()> &strides)
  {
    if constexpr (is_tensor_view_v<OpType>) {
      return op.Slice(starts, ends, strides);
    } else {
      return detail::SliceOp<OpType::Rank(),OpType,decltype(strides)>(op, starts, ends, strides);
    }
  }

  template <typename OpType>
  __MATX_INLINE__ auto slice( const OpType &op,
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends,
      detail::NoStride strides)
  {
    if constexpr (is_tensor_view_v<OpType>) {
      return op.Slice(starts, ends, strides);
    } else {
      return detail::SliceOp<OpType::Rank(),OpType,detail::NoStride>(op, starts, ends, detail::NoStride{});
    }
  }

  template <typename OpType>
  __MATX_INLINE__ auto slice( const OpType &op,
      const index_t (&starts)[OpType::Rank()],
      const index_t (&ends)[OpType::Rank()],
      const index_t (&strides)[OpType::Rank()])
  {
    return slice(op,
        detail::to_array(starts),
        detail::to_array(ends),
        detail::to_array(strides));
  }

  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue.
   *
   * @tparam OpType Input operator/tensor type
   * @param op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @return sliced operator
   */
  template <typename OpType>
  __MATX_INLINE__ auto slice( const OpType &op,
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends)
  {
    return slice(op, starts, ends, detail::NoStride{});
  }
  template <typename OpType>
  __MATX_INLINE__ auto slice( const OpType &op,
      const index_t (&starts)[OpType::Rank()],
      const index_t (&ends)[OpType::Rank()])
  {
    return slice(op,
        detail::to_array(starts),
        detail::to_array(ends));
  }

  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue.
   *
   * The Rank template parameter N is optional when rank does not change
   *
   * @tparam N The Rank of the output operator - optional when slice produces same rank as input
   * @tparam OpType Input operator/tensor type
   * @param op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @param strides Optional:  the stride between consecutive elements
   * @return sliced operator
   */
  template <int N, typename OpType>
    __MATX_INLINE__ auto slice( const OpType &op,
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends,
      const cuda::std::array<index_t, OpType::Rank()> &strides)
  {
    if constexpr (is_tensor_view_v<OpType>) {
      return op.template Slice<N>(starts, ends, strides);
    } else {
      return detail::SliceOp<N,OpType,decltype(strides)>(op, starts, ends, strides);
    }
  }

  template <int N, typename OpType>
    __MATX_INLINE__ auto slice( const OpType &op,
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends,
      [[maybe_unused]] detail::NoStride no_stride)
  {
    if constexpr (is_tensor_view_v<OpType>) {
      return op.template Slice<N>(starts, ends);
    } else {
      return detail::SliceOp<N,OpType,detail::NoStride>(op, starts, ends, no_stride);
    }
  }


  template <int N, typename OpType>
    __MATX_INLINE__ auto slice( const OpType &op,
        const index_t (&starts)[OpType::Rank()],
        const index_t (&ends)[OpType::Rank()],
        const index_t (&strides)[OpType::Rank()])
  {
    return slice<N,OpType>(op,
        detail::to_array(starts),
        detail::to_array(ends),
        detail::to_array(strides));
  }

  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue.

   * The Rank template parameter N is optional when rank does not change
   *
   * @tparam N The Rank of the output operator - optional when slice produces same rank as input
   * @tparam OpType Input operator/tensor type
   * @param op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @return sliced operator
   */
  template <int N, typename OpType>
  __MATX_INLINE__ auto slice (const OpType &op,
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends)
  {
    return slice<N,OpType>(op, starts, ends, detail::NoStride{});
  }

  template <int N, typename OpType>
  __MATX_INLINE__ auto slice (const OpType &op,
      const index_t (&starts)[OpType::Rank()],
      const index_t (&ends)[OpType::Rank()])
  {
    return slice<N,OpType>(op,
        detail::to_array(starts),
        detail::to_array(ends));
  }

#else
   auto slice (const OpType &op,
      const index_t (&starts)[OpType::Rank()],
      const index_t (&ends)[OpType::Rank()]) { }

   auto slice (const OpType &op,
      const index_t (&starts)[OpType::Rank()],
      const index_t (&ends)[OpType::Rank()],
      const index_t (&strides)[OpType::Rank()]) { }
#endif
} // end namespace matx
