/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <misalign.h>

namespace {

template <typename LHS, typename RHS>
__global__ void add_kernel(
  std::size_t volume, legate::Point<3> extents, legate::Point<3> lo, LHS lhs, RHS rhs1, RHS rhs2)
{
  const auto tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid >= volume) {
    return;
  }

  auto idx = tid;

  legate::Point<3> p;
  p[0] = idx / (extents[1] * extents[2]);
  idx %= extents[1] * extents[2];
  p[1] = idx / extents[2];
  idx %= extents[2];
  p[2] = idx;

  p += lo;

  lhs[p] = rhs1[p] + rhs2[p];
}

}  // namespace

void AddTask::gpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM = 3;

  const auto x_input_store = context.input(0);
  const auto y_input_store = context.input(1);
  auto output_store        = context.output(0);

  const auto x_shape      = x_input_store.shape<DIM>();
  const auto y_shape      = y_input_store.shape<DIM>();
  const auto output_shape = output_store.shape<DIM>();

  LEGATE_CHECK(x_shape == y_shape && y_shape == output_shape);

  auto x_acc   = x_input_store.read_accessor<std::int32_t, DIM>();
  auto y_acc   = y_input_store.read_accessor<std::int32_t, DIM>();
  auto out_acc = output_store.write_accessor<std::int32_t, DIM>();

  const auto volume = output_shape.volume();
  auto stream       = context.get_task_stream();

  const auto num_blocks = (volume + 127) / 128;
  const auto extents    = x_shape.hi - x_shape.lo + legate::Point<3>::ONES();

  add_kernel<<<num_blocks, 128, 0, stream>>>(volume, extents, x_shape.lo, out_acc, x_acc, y_acc);
}
