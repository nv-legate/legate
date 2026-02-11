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

namespace legate::cuda::detail {

/**
 * @brief Creates a CUDA event and waits for its work to be completed.
 *
 * This method exists in comparison to CUDA ctx_synchronize
 * because ctx_synchronize will cause the driver to empty its memory pools
 * Then, following cudaMallocAsync calls will re-initialize the pool with
 * a cudaMalloc call, which can be slow. Avoiding this re-initialization is
 * important for performance, thus this alternative.
 */
void sync_current_ctx();

}  // namespace legate::cuda::detail
