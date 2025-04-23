/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>  // LEGATE_NVCC

#include <legate/utilities/macros.h>  // LEGATE_DEFINED

#include <cstdio>
#include <cstdlib>

#if LEGATE_DEFINED(LEGATE_USE_CUDA) || LEGATE_DEFINED(LEGATE_NVCC) || __has_include(<cuda_runtime.h>)
#define LEGATE_CUDA_STUBS 0
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#else  // LEGATE_DEFINED(LEGATE_USE_CUDA)
#include <cstdint>

// Only keep what we need to define LEGATE_CHECK_CUDA and LEGATE_CHECK_CUDA_STREAM

// NOLINTBEGIN
enum cudaError_t : std::int8_t { cudaSuccess };

using CUstream = struct CUstream_st*;

[[nodiscard]] constexpr cudaError_t cudaStreamSynchronize(CUstream) { return cudaSuccess; }

[[nodiscard]] constexpr cudaError_t cudaPeekAtLastError() { return cudaSuccess; }

[[nodiscard]] constexpr const char* cudaGetErrorString(cudaError_t) { return "unknown CUDA error"; }

[[nodiscard]] constexpr const char* cudaGetErrorName(cudaError_t) { return "unknown CUDA error"; }

// NOLINTEND
#endif  // LEGATE_DEFINED(LEGATE_USE_CUDA)

// Use of __CUDACC__ vs LEGATE_USE_CUDA or LEGATE_NVCC is deliberate here, we only want these
// defined when compiling kernels

#define LEGATE_THREADS_PER_BLOCK 128
#define LEGATE_MIN_CTAS_PER_SM 4
#define LEGATE_MAX_REDUCTION_CTAS 1024
#define LEGATE_WARP_SIZE 32
#define LEGATE_CHECK_CUDA(...)                                                 \
  LEGATE_DEPRECATED_MACRO("please roll your own version of LEGATE_CHECK_CUDA") \
  do {                                                                         \
    const cudaError_t legate_cuda_error_result_ = __VA_ARGS__;                 \
    legate::cuda::check_cuda(legate_cuda_error_result_, __FILE__, __LINE__);   \
  } while (false)
// NOLINTNEXTLINE
#define LegateCheckCUDA(...) LEGATE_CHECK_CUDA(__VA_ARGS__)

#if LEGATE_DEFINED(LEGATE_USE_DEBUG)
#define LEGATE_CHECK_CUDA_STREAM(stream)                                              \
  LEGATE_DEPRECATED_MACRO("please roll your own version of LEGATE_CHECK_CUDA_STREAM") \
  do {                                                                                \
    LEGATE_CHECK_CUDA(cudaStreamSynchronize(stream));                                 \
    LEGATE_CHECK_CUDA(cudaPeekAtLastError());                                         \
  } while (false)
#else
#define LEGATE_CHECK_CUDA_STREAM(stream)                                              \
  LEGATE_DEPRECATED_MACRO("please roll your own version of LEGATE_CHECK_CUDA_STREAM") \
  LEGATE_CHECK_CUDA(cudaPeekAtLastError())
#endif
// NOLINTNEXTLINE
#define LegateCheckCUDAStream(...) LEGATE_CHECK_CUDA_STREAM(__VA_ARGS__)

namespace legate::cuda {

LEGATE_HOST inline void check_cuda(cudaError_t error, const char* file, int line)
{
  if (error != cudaSuccess) {
    static_cast<void>(
      std::fprintf(stderr,
                   "Internal CUDA failure with error %s (%s) in file %s at line %d\n",
                   cudaGetErrorString(error),
                   cudaGetErrorName(error),
                   file,
                   line));
    std::abort();
  }
}

}  // namespace legate::cuda
