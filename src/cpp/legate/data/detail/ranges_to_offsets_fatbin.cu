/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/typedefs.h>

#include <cstddef>

namespace detail {

namespace {

__device__ __forceinline__ std::size_t global_tid_1d()
{
  return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

}  // namespace

}  // namespace detail

extern "C" __global__ void legate_ranges_to_offsets_kernel(
  std::size_t ranges_volume,
  legate::Point<1> ranges_lo,
  legate::AccessorWO<std::int32_t, 1> offsets_acc,
  legate::AccessorRO<legate::Rect<1>, 1> ranges_acc)
{
  if (const auto tid = detail::global_tid_1d(); tid < ranges_volume) {
    const auto p = ranges_lo + tid;

    offsets_acc[p] = ranges_acc[p].lo[0] - ranges_acc[ranges_lo].lo[0];
  }
}
