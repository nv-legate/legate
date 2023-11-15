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

#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#include "core/mapping/machine.h"
#include "core/runtime/detail/runtime.h"
#include "core/runtime/runtime.h"

namespace legate::cuda {

StreamView::~StreamView()
{
  if (valid_ && detail::Config::synchronize_stream_view) {
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      CHECK_CUDA_STREAM(stream_);
    } else {
      CHECK_CUDA(cudaStreamSynchronize(stream_));
    }
  }
}

StreamView::StreamView(StreamView&& rhs) : valid_(rhs.valid_), stream_(rhs.stream_)
{
  rhs.valid_ = false;
}

StreamView& StreamView::operator=(StreamView&& rhs)
{
  valid_     = rhs.valid_;
  stream_    = rhs.stream_;
  rhs.valid_ = false;
  return *this;
}

StreamPool::~StreamPool()
{
  if (cached_stream_ != nullptr) {
    CHECK_CUDA(cudaStreamDestroy(*cached_stream_));
  }
}

StreamView StreamPool::get_stream()
{
  if (nullptr == cached_stream_) {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cached_stream_ = std::make_unique<cudaStream_t>(stream);
  }
  return StreamView(*cached_stream_);
}

/*static*/ StreamPool& StreamPool::get_stream_pool()
{
  static StreamPool pools[LEGION_MAX_NUM_PROCS];
  const auto proc = Legion::Processor::get_executing_processor();
  auto proc_id    = proc.id & (LEGION_MAX_NUM_PROCS - 1);
  return pools[proc_id];
}

}  // namespace legate::cuda
