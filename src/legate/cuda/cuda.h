/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "legate_defines.h"

#include "legate/utilities/compiler.h"
#include "legate/utilities/macros.h"

#include <cstdio>
#include <cstdlib>

#if LEGATE_DEFINED(LEGATE_USE_CUDA) || LEGATE_DEFINED(LEGATE_NVCC) || __has_include(<cuda_runtime.h>)
#define LEGATE_CUDA_STUBS 0
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#else  // LEGATE_DEFINED(LEGATE_USE_CUDA)
#include <cstddef>
#include <cstdint>

#define LEGATE_CUDA_STUBS 1

// NOLINTBEGIN(readability-identifier-naming)
constexpr int cudaStreamNonBlocking = 0;

enum cudaMemcpyKind : std::int8_t {
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice,
  cudaMemcpyHostToDevice
};

enum cudaMemoryType : std::int8_t { cudaMemoryTypeDevice, cudaMemoryTypeUnregistered };

enum cudaError_t : std::int8_t { cudaSuccess };

struct CUstream_st;

using cudaStream_t = struct CUstream_st*;

struct cudaPointerAttributes {
  cudaMemoryType type{};
};

// ==========================================================================================

[[nodiscard]] constexpr cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }

[[nodiscard]] constexpr cudaError_t cudaPeekAtLastError() { return cudaSuccess; }

[[nodiscard]] constexpr const char* cudaGetErrorString(cudaError_t) { return "unknown CUDA error"; }

[[nodiscard]] constexpr const char* cudaGetErrorName(cudaError_t) { return "unknown CUDA error"; }

[[nodiscard]] constexpr cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attr,
                                                             const void*)
{
  attr->type = cudaMemoryTypeUnregistered;
  return cudaSuccess;
}

// ==========================================================================================

[[nodiscard]] constexpr cudaError_t cudaMemcpyAsync(
  void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t)
{
  return cudaSuccess;
}

[[nodiscard]] constexpr cudaError_t cudaMemcpy(void*, const void*, std::size_t, cudaMemcpyKind)
{
  return cudaSuccess;
}

// ==========================================================================================

[[nodiscard]] constexpr cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, int)
{
  *stream = cudaStream_t{};
  return cudaSuccess;
}

[[nodiscard]] constexpr cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }

[[nodiscard]] constexpr cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }

// NOLINTEND(readability-identifier-naming)

#endif  // LEGATE_DEFINED(LEGATE_USE_CUDA)

// Use of __CUDACC__ vs LEGATE_USE_CUDA or LEGATE_NVCC is deliberate here, we only want these
// defined when compiling kernels
#ifdef __CUDACC__
#define LEGATE_HOST __host__
#define LEGATE_DEVICE __device__
#define LEGATE_KERNEL __global__
#else
#define LEGATE_HOST
#define LEGATE_DEVICE
#define LEGATE_KERNEL
#endif

#define LEGATE_HOST_DEVICE LEGATE_HOST LEGATE_DEVICE

#define LEGATE_THREADS_PER_BLOCK 128
#define LEGATE_MIN_CTAS_PER_SM 4
#define LEGATE_MAX_REDUCTION_CTAS 1024
#define LEGATE_WARP_SIZE 32
#define LEGATE_CHECK_CUDA(...)                                               \
  do {                                                                       \
    const cudaError_t legate_cuda_error_result_ = __VA_ARGS__;               \
    legate::cuda::check_cuda(legate_cuda_error_result_, __FILE__, __LINE__); \
  } while (false)
// NOLINTNEXTLINE
#define LegateCheckCUDA(...)                                                                       \
  LEGATE_DEPRECATED_MACRO(                                                                         \
    "please roll your own, or, if you must, temporarily use LEGATE_CHECK_CUDA instead. Note that " \
    "LEGATE_CHECK_CUDA will also be removed at some point in the future.")                         \
  LEGATE_CHECK_CUDA(__VA_ARGS__)

#if LEGATE_DEFINED(LEGATE_USE_DEBUG)
#define LEGATE_CHECK_CUDA_STREAM(stream)              \
  do {                                                \
    LEGATE_CHECK_CUDA(cudaStreamSynchronize(stream)); \
    LEGATE_CHECK_CUDA(cudaPeekAtLastError());         \
  } while (false)
#else
#define LEGATE_CHECK_CUDA_STREAM(stream) static_cast<void>(stream)
#endif
// NOLINTNEXTLINE
#define LegateCheckCUDAStream(...)                                                              \
  LEGATE_DEPRECATED_MACRO(                                                                      \
    "please roll your own, or, if you must, temporarily use LEGATE_CHECK_CUDA_STREAM instead. " \
    "Note that LEGATE_CHECK_CUDA_STREAM will also be removed at some point in the future.")     \
  LEGATE_CHECK_CUDA_STREAM(__VA_ARGS__)

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
