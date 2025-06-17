/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <cstdint>

namespace detail {

namespace {

__device__ __forceinline__ std::size_t global_tid_1d()
{
  return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

}  // namespace

}  // namespace detail

extern "C" __global__ void legate_offsets_to_ranges_kernel(
  std::size_t offsets_volume,
  std::int64_t vardata_volume,
  legate::Point<1> offsets_lo,
  legate::Point<1> vardata_lo,
  legate::AccessorWO<legate::Rect<1>, 1> ranges_acc,
  legate::AccessorRO<std::int32_t, 1> offsets_acc)
{
  if (const auto tid = detail::global_tid_1d(); tid >= offsets_volume) {
    const auto p = offsets_lo + tid;
    auto& range  = ranges_acc[p];

    range.lo[0] = vardata_lo + offsets_acc[p];
    range.hi[0] =
      vardata_lo + (tid != offsets_volume - 1 ? offsets_acc[p + 1] : vardata_volume) - 1;
  }
}
