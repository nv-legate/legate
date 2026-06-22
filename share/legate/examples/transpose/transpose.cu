/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <transpose.h>

void CopyTask::gpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM = 2;
  const auto input           = context.input(0);
  const auto output          = context.output(0);
  const auto shape           = input.shape<DIM>();
  const auto in_acc          = input.read_accessor<std::int32_t, DIM>();
  auto out_acc               = output.write_accessor<std::int32_t, DIM>();

  auto in_ptr  = input.read_accessor<std::int32_t, DIM>().ptr(shape);
  auto out_ptr = output.write_accessor<std::int32_t, DIM>().ptr(shape);

  auto stream = context.get_task_stream();

  auto status = cudaMemcpyAsync(
    out_ptr, in_ptr, shape.volume() * sizeof(std::int32_t), cudaMemcpyDefault, stream);

  LEGATE_CHECK(status == cudaSuccess);
}
