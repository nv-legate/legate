/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <cstdint>

namespace legate::detail {

namespace {

__device__ __forceinline__ void fill_(std::uint64_t* store, std::size_t len, std::uint64_t value)
{
  auto idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx >= len) {
    return;
  }

  store[idx] = value;
}

}  // namespace

// sizes 32 or less can be handled by cuMemset variants
extern "C" __global__ void legate_fill_kernel_64(std::uint64_t* store,
                                                 std::size_t len,
                                                 std::uint64_t value)
{
  fill_(store, len, value);
}

}  // namespace legate::detail
