/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#define LEGATE_SILENCE_STREAM_POOL_DEPRECATION_PRIVATE 1

#include <legate/cuda/stream_pool.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/runtime/detail/config.h>
#include <legate/runtime/detail/runtime.h>

namespace legate::cuda {

StreamView::~StreamView()
{
  if (valid_ && legate::detail::Config::get_config().synchronize_stream_view()) {
    detail::get_cuda_driver_api()->stream_synchronize(stream_);
  }
}

StreamView StreamPool::get_stream()
{
  return StreamView{legate::detail::Runtime::get_runtime().get_cuda_stream()};
}

/*static*/ StreamPool& StreamPool::get_stream_pool()
{
  static StreamPool pool;
  return pool;
}

}  // namespace legate::cuda
