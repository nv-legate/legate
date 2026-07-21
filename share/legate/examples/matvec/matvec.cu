/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <matvec.h>

namespace {

// This kernel is horribly inefficient
template <typename IN_ACC, typename RED_ACC>
__global__ void matvec_kernel(std::size_t volume,
                              IN_ACC rhs1,
                              IN_ACC rhs2,
                              RED_ACC lhs,
                              legate::NDPoint<2> lo,
                              std::int64_t stride)
{
  const auto idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx >= volume) {
    return;
  }

  auto p = lo;

  p[0] += idx / stride;
  p[1] += idx % stride;

  lhs.reduce(p, rhs1[p] * rhs2[p]);
}

}  // namespace

void MatVecTask::gpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM                = 2;
  constexpr std::uint32_t THREAD_BLOCK_SIZE = 128;

  const auto rhs1     = context.input(0);
  const auto rhs2     = context.input(1);
  const auto lhs      = context.reduction(0);
  const auto shape    = rhs1.shape<DIM>();
  const auto rhs1_acc = rhs1.read_accessor<std::int32_t, DIM>();
  const auto rhs2_acc = rhs2.read_accessor<std::int32_t, DIM>();
  auto lhs_acc =
    lhs.reduce_accessor<legate::SumReduction<std::int32_t>, /*exclusive=*/false, DIM>();

  const auto volume     = shape.volume();
  const auto num_blocks = (volume + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

  matvec_kernel<<<num_blocks, THREAD_BLOCK_SIZE, 0, context.get_task_stream()>>>(
    volume, rhs1_acc, rhs2_acc, lhs_acc, shape.lo, shape.hi[1] - shape.lo[1] + 1);
}
