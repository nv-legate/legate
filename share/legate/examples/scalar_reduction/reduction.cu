/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <reduction.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

/*
void ReductionTask::gpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM = 2;

  const auto x_input_store = context.input(0);
  const auto y_input_store = context.input(1);
  auto output_store        = context.output(0);

  const auto x_shape      = x_input_store.shape<DIM>();
  const auto y_shape      = y_input_store.shape<DIM>();
  const auto output_shape = output_store.shape<DIM>();

  LEGATE_CHECK(x_shape == y_shape && y_shape == output_shape);

  auto x_ptr      = x_input_store.read_accessor<std::int32_t, DIM>().ptr(x_shape);
  auto y_ptr      = y_input_store.read_accessor<std::int32_t, DIM>().ptr(y_shape);
  auto output_ptr = output_store.write_accessor<std::int32_t, DIM>().ptr(output_shape);

  const auto volume = output_shape.volume();
  auto stream       = context.get_task_stream();

  auto begin = thrust::make_zip_iterator(thrust::make_tuple(x_ptr, y_ptr));
  auto end   = thrust::make_zip_iterator(thrust::make_tuple(x_ptr + volume, y_ptr + volume));

  thrust::transform(thrust::cuda::par.on(stream), begin, end, output_ptr, AddFunctor{});
}
*/
