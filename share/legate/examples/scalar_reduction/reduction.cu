/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/redop/base.h>

#include <reduction.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>

template <typename ACCESSOR>
__global__ void reduce_kernel(ACCESSOR target, std::uint64_t value)
{
  target.reduce(0, value);
}

void ReductionTask::gpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM = 1;

  const auto input = context.input(0);
  auto output      = context.reduction(0);

  const auto shape = input.shape<DIM>();

  auto stream  = context.get_task_stream();
  auto in_acc  = input.read_accessor<std::uint64_t, DIM>();
  auto in_ptr  = in_acc.ptr(shape.lo);
  auto red_acc = output.reduce_accessor<legate::SumReduction<std::uint64_t>, true, DIM>();

  auto out = thrust::reduce(thrust::cuda::par.on(stream),
                            in_ptr,
                            in_ptr + shape.volume(),
                            std::uint64_t{0},
                            thrust::plus<std::uint64_t>{});

  reduce_kernel<<<1, 1, 0, stream>>>(red_acc, out);
}
