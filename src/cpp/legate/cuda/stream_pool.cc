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

#include "legate/cuda/detail/cuda_driver_api.h"
#include "legate/runtime/detail/config.h"
#include "legate/runtime/detail/runtime.h"

namespace legate::cuda {

namespace {

[[nodiscard]] const detail::CUDADriverAPI* get_driver()
{
  return legate::detail::Runtime::get_runtime()->get_cuda_driver_api();
}

}  // namespace

StreamView::~StreamView()
{
  if (valid_ && legate::detail::Config::synchronize_stream_view) {
    LEGATE_CHECK_CUDRIVER(get_driver()->stream_synchronize(stream_));
  }
}

StreamPool::~StreamPool()
{
  if (cached_stream_) {
    LEGATE_CHECK_CUDRIVER(get_driver()->stream_destroy(*cached_stream_));
  }
}

StreamView StreamPool::get_stream()
{
  if (!cached_stream_.has_value()) {
    CUstream stream;

    LEGATE_CHECK_CUDRIVER(get_driver()->stream_create(&stream, CU_STREAM_NON_BLOCKING));
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
