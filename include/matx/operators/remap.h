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
   * Remaps elements an operator according to an index array/operator.
   */
  namespace detail {
    template <int DIM, typename T, typename IdxType>
      class RemapOp : public BaseOp<RemapOp<DIM, T, IdxType>>
    {
      private:
        //mutable typename detail::base_type_t<T> op_;
        typename detail::base_type_t<T> op_;
        typename detail::base_type_t<IdxType> idx_;

      public:
        using matxop = bool;
        using matxoplvalue = bool;

        using value_type = typename T::value_type;
        using index_type = typename IdxType::value_type;
        using self_type = RemapOp<DIM, T, IdxType>;
        static_assert(std::is_integral<index_type>::value, "RemapOp: Type for index operator must be integral");
        static_assert(IdxType::Rank() <= 1, "RemapOp: Rank of index operator must be 0 or 1");
        static_assert(DIM<T::Rank(), "RemapOp: DIM must be less than Rank of tensor");

        __MATX_INLINE__ std::string str() const { return "remap(" + op_.str() + ")"; }

	      __MATX_INLINE__ RemapOp(const T &op, IdxType idx) : op_(op), idx_(idx) {};

        // Only supports one element per thread
        template <ElementsPerThread EPT, typename Op, typename Idx, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, const Idx &idx, Is... indices)
        {
          if constexpr (EPT == ElementsPerThread::ONE) {
            static_assert(sizeof...(Is) == Rank());
            static_assert((std::is_convertible_v<Is, index_t> && ... ));

            cuda::std::array ind{indices...};

            // remap current index for dim
            if constexpr (IdxType::Rank() == 0) {
              ind[DIM] = get_value<EPT>(idx);
            } else {
              ind[DIM] = get_value<EPT>(idx, ind[DIM]);
            }

            return get_value<EPT>(cuda::std::forward<Op>(op), ind);

          } else {
            return Vector<value_type, static_cast<size_t>(EPT)>{};
          }
        }

        template <ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return get_impl<EPT>(cuda::std::as_const(op_), idx_, indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<detail::ElementsPerThread::ONE>(indices...);
        }

        template <ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return get_impl<EPT>(cuda::std::forward<decltype(op_)>(op_), idx_, indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return this->operator()<detail::ElementsPerThread::ONE>(indices...);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return T::Rank();
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int32_t dim) const
        {
          if(dim == DIM) {
            if constexpr (IdxType::Rank() == 0) {
              return 1;
            } else {
              return idx_.Size(0);
            }
          } else {
            return op_.Size(dim);
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

        template <OperatorCapability Cap>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            return ElementsPerThread::ONE;
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(
              self_has_cap,
              detail::get_operator_capability<Cap>(op_),
              detail::get_operator_capability<Cap>(idx_)
            );
          }
        }

        ~RemapOp() = default;
        RemapOp(const RemapOp &rhs) = default;
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
    };
  }

  /**
   * @brief Operator to logically remap elements of an operator based on an index array/operator.
   *
   * The rank of the output tensor is equal to the rank of the input tensor.
   * The rank of the index tensor must be 0 or 1.
   *
   * The size of the output tensor is the same as the input tensor except in the applied dimenions.
   * In the applied dimension the size of the output tensor is equal to the size of the index tensor.
   * In the case of a 0-rank index tensor, the size of the output tensor in the corresponding
   * dimension is always 1.
   *
   * This operator can appear as an rvalue or lvalue.
   *
   * @tparam DIM Dimension to apply the remap
   * @tparam T Input operator/tensor type
   * @tparam Ind Input index Operator type
   * @param t Input operator
   * @param idx Index operator/tensor
   * @return Value in t from each location in idx
   */
  template <int DIM, typename Op, typename Ind>
    auto __MATX_INLINE__ remap(const Op &t, Ind idx)
    {
      return detail::RemapOp<DIM, Op, Ind>(t, idx);
    };

  /**
   * @brief Operator to logically remap elements of an operator based on an index array/operator.
   *
   * The rank of the output tensor is equal to the rank of the input tensor.
   * The rank of the index tensor must be 0 or 1.
   * The number of DIMS and the number of Inds provided must be the same.
   *
   * The size of the output tensor is the same as the input tensor except in the applied dimenions.
   * In the applied dimension the size of the output tensor is equal to the size of the index tensor.
   * In the case of a 0-rank index tensor, the size of the output tensor in the corresponding
   * dimension is always 1.
   *
   * This operator can appear as an rvalue or lvalue.
   *
   * @tparam DIM Dimension to apply the remap
   * @tparam DIMS... list of multiple dimensions to remap along
   * @tparam T Input operator/tensor type
   * @tparam Ind Input index Operator type
   * @tparam Inds... list of multiple index operators to remap along
   * @param t Input operator
   * @param idx Index operator/tensor
   * @param inds list of multiple index operators to remap along
   * @return Value in t from each location in idx
   */
  template <int DIM, int... DIMS, typename Op, typename Ind, typename... Inds>
    auto __MATX_INLINE__ remap(const Op &t, Ind idx, Inds... inds)
    {
      static_assert(sizeof...(DIMS) == sizeof...(Inds), "remap number of DIMs must match number of index arrays");

      // recursively call remap on remaining bits
      auto op = remap<DIMS...>(t, inds...);

      // construct remap op
      return detail::RemapOp<DIM, decltype(op) , Ind>(op, idx);
    };
} // end namespace matx
