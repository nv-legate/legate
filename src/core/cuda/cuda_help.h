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

#include "legion.h"

#include "legate_defines.h"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 128
#define CHECK_CUDA(...)                                                      \
  do {                                                                       \
    const cudaError_t legate_cuda_error_result_ = __VA_ARGS__;               \
    legate::cuda::check_cuda(legate_cuda_error_result_, __FILE__, __LINE__); \
  } while (false)

#if LegateDefined(LEGATE_USE_DEBUG)
#define CHECK_CUDA_STREAM(stream)              \
  do {                                         \
    CHECK_CUDA(cudaStreamSynchronize(stream)); \
    CHECK_CUDA(cudaPeekAtLastError());         \
  } while (false)
#else
#define CHECK_CUDA_STREAM(stream)
#endif

namespace legate::cuda {

__host__ inline void check_cuda(cudaError_t error, const char* file, int line)
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
