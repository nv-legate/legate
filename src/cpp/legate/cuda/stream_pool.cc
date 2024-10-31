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

StreamView::~StreamView()
{
  if (valid_ && legate::detail::Config::synchronize_stream_view) {
    LEGATE_CHECK_CUDRIVER(
      legate::detail::Runtime::get_runtime()->get_cuda_driver_api()->stream_synchronize(stream_));
  }
}

StreamView StreamPool::get_stream()
{
  return StreamView{legate::detail::Runtime::get_runtime()->get_cuda_stream()};
}

/*static*/ StreamPool& StreamPool::get_stream_pool()
{
  static StreamPool pool;
  return pool;
}

}  // namespace legate::cuda
