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

#include "core/utilities/compiler.h"
#include "core/utilities/macros.h"

#include "legate_defines.h"

#include <cstdio>
#include <cstdlib>

#if LegateDefined(LEGATE_USE_CUDA) || LegateDefined(LEGATE_NVCC)
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#else  // LegateDefined(LEGATE_USE_CUDA)
#include <cstddef>
#include <cstdint>

#define LEGATE_CUDA_STUBS 1

constexpr int cudaStreamNonBlocking = 0;

enum cudaMemcpyKind : std::int8_t {
  cudaMemcpyDeviceToHost,
  cudaMemcpyDeviceToDevice,
  cudaMemcpyHostToDevice
};

enum cudaMemoryType : std::int8_t { cudaMemoryTypeDevice, cudaMemoryTypeUnregistered };

enum cudaError_t : std::int8_t { cudaSuccess };

using cudaStream_t = struct cudaStream_st*;

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

#endif  // LegateDefined(LEGATE_USE_CUDA)

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
#define LegateCheckCUDA(...)                                                 \
  do {                                                                       \
    const cudaError_t legate_cuda_error_result_ = __VA_ARGS__;               \
    legate::cuda::check_cuda(legate_cuda_error_result_, __FILE__, __LINE__); \
  } while (false)

#if LegateDefined(LEGATE_USE_DEBUG)
#define LegateCheckCUDAStream(stream)               \
  do {                                              \
    LegateCheckCUDA(cudaStreamSynchronize(stream)); \
    LegateCheckCUDA(cudaPeekAtLastError());         \
  } while (false)
#else
#define LegateCheckCUDAStream(stream) static_cast<void>(stream)
#endif

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
