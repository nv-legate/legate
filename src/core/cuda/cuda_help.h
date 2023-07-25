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

#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(expr)                                      \
  do {                                                        \
    cudaError_t __result__ = (expr);                          \
    legate::cuda::check_cuda(__result__, __FILE__, __LINE__); \
  } while (false)

#ifdef DEBUG_LEGATE

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
    fprintf(stderr,
            "Internal CUDA failure with error %s (%s) in file %s at line %d\n",
            cudaGetErrorString(error),
            cudaGetErrorName(error),
            file,
            line);
#ifdef DEBUG_LEGATE
    assert(false);
#else
    exit(error);
#endif
  }
}

}  // namespace legate::cuda
