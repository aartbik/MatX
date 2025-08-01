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
   * Repeats a matrix the specified amount of times
   *
   * RepMatOp performs a "repmat" operation on a matrix where each dimension
   * specified in "reps" is repeated. Constructors for both scalars and arrays are
   * provided. The scalar version will repeat the matrix by the scalar amount in
   * every dimension, whereas the array version scales independently by each
   * dimension.
   */
  namespace detail {
    template <typename T1, int DIM>
      class RepMatOp : public BaseOp<RepMatOp<T1, DIM>>
    {
      private:
        typename detail::base_type_t<T1> op_;
        index_t reps_[T1::Rank()];

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

        __MATX_INLINE__ std::string str() const { return "repmat(" + op_.str() + ")"; }

          __MATX_INLINE__ RepMatOp(const T1 &op, index_t reps) : op_(op)
        {
          for (int dim = 0; dim < DIM; dim++)
          {
            reps_[dim] = reps;
          }
        }

        __MATX_INLINE__ RepMatOp(const T1 &op, const cuda::std::array<index_t, DIM> reps) : op_(op)
        {
          for (int dim = 0; dim < DIM; dim++)
          {
            reps_[dim] = reps[dim];
          }
        }

        __MATX_INLINE__ RepMatOp(const T1 &op, const index_t *reps) : op_(op)
        {
          for (int dim = 0; dim < DIM; dim++)
          {
            reps_[dim] = reps[dim];
          }
        }

        template <ElementsPerThread EPT, typename Op, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, Is... indices)
        {
          if constexpr (EPT == ElementsPerThread::ONE) {
            if constexpr (Rank() == 0) {
              return op.template operator()<EPT>();
            }
            else {
              cuda::std::array idx{indices...};

              MATX_LOOP_UNROLL
              for (int i = 0; i < static_cast<int>(idx.size()); i++) {
                idx[i] %= op.Size(i);
              }

              return get_value<EPT>(cuda::std::forward<Op>(op), idx);
            }
          } else {
            return Vector<value_type, static_cast<index_t>(EPT)>{};
          }
        }

        template <ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return get_impl<EPT>(cuda::std::as_const(op_), indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<detail::ElementsPerThread::ONE>(indices...);
        }

        template <ElementsPerThread EPT, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return get_impl<EPT>(cuda::std::forward<decltype(op_)>(op_), indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return this->operator()<detail::ElementsPerThread::ONE>(indices...);
        }

        template <OperatorCapability Cap>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability() const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            return ElementsPerThread::ONE;
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::get_rank<T1>();
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return op_.Size(dim) * reps_[dim];
        }
    };
  }

  /**
   * Repeat a matrix an equal number of times in each dimension
   *
   * @tparam T1
   *   Type of operator or view
   * @param t
   *   Operator or view to repeat
   * @param reps
   *   Amount to repeat
   *
   * @returns
   *   New operator with repeated data
   */
  template <typename T1>
    auto __MATX_INLINE__ repmat(const T1 &t, index_t reps)
    {
      return detail::RepMatOp<T1, T1::Rank()>(t, reps);
    };

  /**
   * Repeat a matrix a specific number of times in each direction
   *
   * @tparam T1
   *   Type of operator or view
   * @param t
   *   Operator or view to repeat
   * @param reps
   *   Array of times to repeat in each dimension
   *
   * @returns
   *   New operator with repeated data
   */
  template <typename T1, int N>
    auto __MATX_INLINE__ repmat(const T1 &t, const index_t (&reps)[N])
    {
      return detail::RepMatOp<T1, T1::Rank()>(t, detail::to_array(reps));
    };

  /**
   * Repeat a matrix a specific number of times in each direction
   *
   * @tparam T1
   *   Type of operator or view
   * @param t
   *   Operator or view to repeat
   * @param reps
   *   Array of times to repeat in each dimension
   *
   * @returns
   *   New operator with repeated data
   */
  template <typename T1>
    auto __MATX_INLINE__ repmat(const T1 &t, const index_t *reps)
    {
      return detail::RepMatOp<T1, T1::Rank()>(t, reps);
    };

} // end namespace matx
