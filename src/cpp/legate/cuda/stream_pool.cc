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

#define LEGATE_SILENCE_STREAM_POOL_DEPRECATION_PRIVATE 1

#include "legate/cuda/stream_pool.h"

#include "legate/runtime/detail/config.h"
#include "legate/runtime/detail/runtime.h"

namespace legate::cuda {

StreamView::~StreamView()
{
  if (valid_) {
    if (LEGATE_DEFINED(LEGATE_USE_DEBUG) || legate::detail::Config::synchronize_stream_view) {
      LEGATE_CHECK_CUDA(cudaStreamSynchronize(stream_));
      LEGATE_CHECK_CUDA(cudaPeekAtLastError());
    } else {
      LEGATE_CHECK_CUDA(cudaPeekAtLastError());
    }
  }
}

StreamPool::~StreamPool()
{
  if (cached_stream_) {
    LEGATE_CHECK_CUDA(cudaStreamDestroy(*cached_stream_));
  }
}

StreamView StreamPool::get_stream()
{
  if (!cached_stream_.has_value()) {
    cudaStream_t stream;

    LEGATE_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cached_stream_.emplace(stream);
  }
  return StreamView{*cached_stream_};
}

/*static*/ StreamPool& StreamPool::get_stream_pool()
{
  static StreamPool pools[LEGION_MAX_NUM_PROCS];
  const auto proc = legate::detail::Runtime::get_runtime()->get_executing_processor();
  auto proc_id    = proc.id & (LEGION_MAX_NUM_PROCS - 1);
  return pools[proc_id];
}

}  // namespace legate::cuda
