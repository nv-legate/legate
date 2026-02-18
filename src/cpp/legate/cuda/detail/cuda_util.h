/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#define LEGATE_THREADS_PER_BLOCK 128
#define LEGATE_MIN_CTAS_PER_SM 4
#define LEGATE_MAX_REDUCTION_CTAS 1024
#define LEGATE_WARP_SIZE 32

#include <legate/cuda/detail/cuda_driver_api.h>

namespace legate::cuda::detail {

/**
 * @brief Creates a CUDA event and waits for work on the context to be completed.
 *
 * This method exists in comparison to CUDA ctx_synchronize
 * because ctx_synchronize will cause the driver to empty its memory pools
 * Then, following cudaMallocAsync calls will re-initialize the pool with
 * a cudaMalloc call, which can be slow. Avoiding this re-initialization is
 * important for performance, thus this alternative.
 */
void sync_current_ctx();

/**
 * @brief Creates a CUDA event and waits for work on the current stream to be completed.
 *
 * This method only synchronizes the current stream, not the current context,
 * making it a lighter alternative to sync_current_ctx.
 *
 * This method exists in comparison to CUDA stream_synchronize
 * because stream_synchronize may cause the driver to empty its memory pools
 * Then, following cudaMallocAsync calls will re-initialize the pool with
 * a cudaMalloc call, which can be slow. Avoiding this re-initialization is
 * important for performance, thus this alternative.
 *
 * @param stream The stream to synchronize.
 */
void stream_synchronize_minimal(CUstream stream);

}  // namespace legate::cuda::detail
