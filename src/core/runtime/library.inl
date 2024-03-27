/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

// Useful for IDEs
#include "core/runtime/library.h"
#include "core/utilities/typedefs.h"

namespace legate::detail {

#ifndef REALM_COMPILER_IS_NVCC

template <typename REDOP>
void register_reduction_callback(const Legion::RegistrationCallbackArgs& args)
{
  auto legion_redop_id = *static_cast<const int32_t*>(args.buffer.get_ptr());
  Legion::Runtime::register_reduction_op<REDOP>(legion_redop_id);
}

#else  // ifndef REALM_COMPILER_IS_NVCC

template <typename T>
class CUDAReductionOpWrapper : public T {
 public:
  static const bool has_cuda_reductions = true;

  template <bool EXCLUSIVE>
  __device__ static void apply_cuda(typename T::LHS& lhs, typename T::RHS rhs)
  {
    T::template apply<EXCLUSIVE>(lhs, rhs);
  }

  template <bool EXCLUSIVE>
  __device__ static void fold_cuda(typename T::RHS& lhs, typename T::RHS rhs)
  {
    T::template fold<EXCLUSIVE>(lhs, rhs);
  }
};

template <typename REDOP>
void register_reduction_callback(const Legion::RegistrationCallbackArgs& args)
{
  auto legion_redop_id = *static_cast<const int32_t*>(args.buffer.get_ptr());
  Legion::Runtime::register_reduction_op(
    legion_redop_id,
    Realm::ReductionOpUntyped::create_reduction_op<detail::CUDAReductionOpWrapper<REDOP>>(),
    nullptr,
    nullptr,
    false);
}

#endif  // ifndef REALM_COMPILER_IS_NVCC

}  // namespace legate::detail

namespace legate {

inline Library::Library(detail::Library* impl) : impl_{impl} {}

template <typename REDOP>
std::int64_t Library::register_reduction_operator(std::int32_t redop_id)
{
  std::int64_t legion_redop_id = get_reduction_op_id(redop_id);
#if !defined(__CUDACC__)
  if (LegateDefined(LEGATE_USE_CUDA)) {
    detail::log_legate().warning() << "For the runtime's DMA engine to GPU accelerate reductions, "
                                      "this reduction operator should be registered in a .cu file.";
  }
#endif
  perform_callback(detail::register_reduction_callback<REDOP>,
                   Legion::UntypedBuffer(&legion_redop_id, sizeof(int32_t)));
  return legion_redop_id;
}

inline bool Library::operator==(const Library& other) const { return impl() == other.impl(); }

inline bool Library::operator!=(const Library& other) const { return !(*this == other); }

inline const detail::Library* Library::impl() const { return impl_; }

inline detail::Library* Library::impl() { return impl_; }

}  // namespace legate
