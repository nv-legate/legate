/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Useful for IDEs
#include <legate_defines.h>

#include <legate/runtime/library.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/typedefs.h>

namespace legate::detail {

template <typename T>
class CUDAReductionOpWrapper : public T {
 public:
  static constexpr bool has_cuda_reductions = true;  // NOLINT(readability-identifier-naming)

  template <bool EXCLUSIVE>
  LEGATE_DEVICE static void apply_cuda(typename T::LHS& lhs, typename T::RHS rhs)
  {
    T::template apply<EXCLUSIVE>(lhs, std::move(rhs));
  }

  template <bool EXCLUSIVE>
  LEGATE_DEVICE static void fold_cuda(typename T::RHS& lhs, typename T::RHS rhs)
  {
    T::template fold<EXCLUSIVE>(lhs, std::move(rhs));
  }
};

template <typename REDOP>
void register_reduction_callback(const Legion::RegistrationCallbackArgs& args)
{
  const auto legion_redop_id = *static_cast<const Legion::ReductionOpID*>(args.buffer.get_ptr());

  if constexpr (LEGATE_DEFINED(LEGATE_NVCC)) {
    Legion::Runtime::register_reduction_op(
      legion_redop_id,
      Realm::ReductionOpUntyped::create_reduction_op<detail::CUDAReductionOpWrapper<REDOP>>(),
      nullptr,
      nullptr,
      false);
  } else {
    Legion::Runtime::register_reduction_op<REDOP>(legion_redop_id);
  }
}

}  // namespace legate::detail

namespace legate {

inline Library::Library(detail::Library* impl) : impl_{impl} {}

template <typename REDOP>
GlobalRedopID Library::register_reduction_operator(LocalRedopID redop_id)
{
  auto legion_redop_id = get_reduction_op_id(redop_id);
#ifndef __CUDACC__
  if (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    detail::log_legate().warning() << "For the runtime's DMA engine to GPU accelerate reductions, "
                                      "this reduction operator should be registered in a .cu file.";
  }
#endif
  perform_callback_(detail::register_reduction_callback<REDOP>,
                    Legion::UntypedBuffer{&legion_redop_id, sizeof(decltype(legion_redop_id))});
  return legion_redop_id;
}

inline bool Library::operator==(const Library& other) const { return impl() == other.impl(); }

inline bool Library::operator!=(const Library& other) const { return !(*this == other); }

inline const detail::Library* Library::impl() const { return impl_; }

inline detail::Library* Library::impl() { return impl_; }

}  // namespace legate
