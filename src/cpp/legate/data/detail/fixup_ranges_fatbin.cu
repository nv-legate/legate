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

#include <legate/utilities/typedefs.h>

#include <cstddef>

namespace {

__device__ __forceinline__ std::size_t global_tid_1d()
{
  return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

}  // namespace

extern "C" __global__ void legate_fixup_ranges_kernel(
  std::size_t desc_volume,
  legate::Point<1> desc_lo,
  legate::Point<1> vardata_lo,
  legate::AccessorRW<legate::Rect<1>, 1> desc_acc)
{
  if (const auto tid = global_tid_1d(); tid < desc_volume) {
    auto& desc = desc_acc[desc_lo + tid];

    desc.lo += vardata_lo;
    desc.hi += vardata_lo;
  }
}
