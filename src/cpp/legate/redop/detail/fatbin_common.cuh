/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <realm/cuda/cuda_redop.h>

#include <cstddef>
#include <cstdint>

namespace legate::detail {

#ifndef DOXYGEN

template <bool EXCL, typename REDOP>
__device__ void apply_cuda_kernel(std::uintptr_t lhs_base,
                                  std::uintptr_t lhs_stride,
                                  std::uintptr_t rhs_base,
                                  std::uintptr_t rhs_stride,
                                  std::size_t count,
                                  REDOP redop)
{
  Realm::Cuda::ReductionKernels::iter_cuda_kernel<typename REDOP::LHS, typename REDOP::RHS>(
    lhs_base,
    lhs_stride,
    rhs_base,
    rhs_stride,
    count,
    Realm::Cuda::ReductionKernels::redop_apply_wrapper<REDOP, EXCL>,
    reinterpret_cast<void*>(&redop));
}

template <bool EXCL, typename REDOP>
__device__ void fold_cuda_kernel(std::uintptr_t rhs1_base,
                                 std::uintptr_t rhs1_stride,
                                 std::uintptr_t rhs2_base,
                                 std::uintptr_t rhs2_stride,
                                 std::size_t count,
                                 REDOP redop)
{
  Realm::Cuda::ReductionKernels::iter_cuda_kernel<typename REDOP::RHS, typename REDOP::RHS>(
    rhs1_base,
    rhs1_stride,
    rhs2_base,
    rhs2_stride,
    count,
    Realm::Cuda::ReductionKernels::redop_fold_wrapper<REDOP, EXCL>,
    reinterpret_cast<void*>(&redop));
}

#define LEGATE_DEFINE_REDOP_STUBS(FUNC_NAME_BASE, TYPE)                                        \
  extern "C" __global__ void FUNC_NAME_BASE##_apply_excl(std::uintptr_t lhs_base,              \
                                                         std::uintptr_t lhs_stride,            \
                                                         std::uintptr_t rhs_base,              \
                                                         std::uintptr_t rhs_stride,            \
                                                         std::size_t count)                    \
  {                                                                                            \
    legate::detail::apply_cuda_kernel<true>(lhs_base,                                          \
                                            lhs_stride,                                        \
                                            rhs_base,                                          \
                                            rhs_stride,                                        \
                                            count, /**/                                        \
                                            TYPE{} /* NOLINT(bugprone-macro-parentheses) */    \
    );                                                                                         \
  }                                                                                            \
                                                                                               \
  extern "C" __global__ void FUNC_NAME_BASE##_apply_non_excl(std::uintptr_t lhs_base,          \
                                                             std::uintptr_t lhs_stride,        \
                                                             std::uintptr_t rhs_base,          \
                                                             std::uintptr_t rhs_stride,        \
                                                             std::size_t count)                \
  {                                                                                            \
    legate::detail::apply_cuda_kernel<false>(lhs_base,                                         \
                                             lhs_stride,                                       \
                                             rhs_base,                                         \
                                             rhs_stride,                                       \
                                             count, /**/                                       \
                                             TYPE{} /* NOLINT(bugprone-macro-parentheses) */); \
  }                                                                                            \
                                                                                               \
  extern "C" __global__ void FUNC_NAME_BASE##_fold_excl(std::uintptr_t lhs_base,               \
                                                        std::uintptr_t lhs_stride,             \
                                                        std::uintptr_t rhs_base,               \
                                                        std::uintptr_t rhs_stride,             \
                                                        std::size_t count)                     \
  {                                                                                            \
    legate::detail::fold_cuda_kernel<true>(lhs_base,                                           \
                                           lhs_stride,                                         \
                                           rhs_base,                                           \
                                           rhs_stride,                                         \
                                           count, /**/                                         \
                                           TYPE{} /* NOLINT(bugprone-macro-parentheses) */     \
    );                                                                                         \
  }                                                                                            \
                                                                                               \
  extern "C" __global__ void FUNC_NAME_BASE##_fold_non_excl(std::uintptr_t lhs_base,           \
                                                            std::uintptr_t lhs_stride,         \
                                                            std::uintptr_t rhs_base,           \
                                                            std::uintptr_t rhs_stride,         \
                                                            std::size_t count)                 \
  {                                                                                            \
    /* NOLINTBEGIN(clang-analyzer-core.uninitialized.UndefReturn) */                           \
    legate::detail::fold_cuda_kernel<false>(lhs_base,                                          \
                                            lhs_stride,                                        \
                                            rhs_base,                                          \
                                            rhs_stride,                                        \
                                            count, /**/                                        \
                                            TYPE{} /* NOLINT(bugprone-macro-parentheses) */    \
    );                                                                                         \
    /* NOLINTEND(clang-analyzer-core.uninitialized.UndefReturn) */                             \
  }                                                                                            \
  static_assert(true)

#endif

}  // namespace legate::detail
