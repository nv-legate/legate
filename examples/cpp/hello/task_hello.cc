/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "task_hello.h"

namespace task {

namespace hello {

Legion::Logger logger(library_name);

void register_tasks()
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->create_library(library_name);
  HelloWorldTask::register_variants(library);
  IotaTask::register_variants(library);
  SquareTask::register_variants(library);
  SumTask::register_variants(library);
}

/*static*/ void HelloWorldTask::cpu_variant(legate::TaskContext context)
{
  std::string message = context.scalar(0).value<std::string>();
  std::cout << message << std::endl;
}

/*static*/ void SumTask::cpu_variant(legate::TaskContext context)
{
  legate::Store input         = context.input(0);
  legate::Rect<1> input_shape = input.shape<1>();  // should be a 1-Dim array
  auto in                     = input.read_accessor<float, 1>();

  logger.info() << "Sum [" << input_shape.lo << "," << input_shape.hi << "]";

  float total = 0;
  // i is a global index for the complete array
  for (int64_t i = input_shape.lo; i <= input_shape.hi; ++i) { total += in[i]; }

  /**
    The task launch as a whole will return a single value (Store of size 1)
    to the caller. However, each point task gets a separate Store of the
    same size as the result, to reduce into. This "local accumulator" will
    be initialized by the runtime, and all we need to do is call .reduce()
    to add our local contribution. After all point tasks return, the runtime
    will make sure to combine all their buffers into the single final result.
  */
  using Reduce         = Legion::SumReduction<float>;
  legate::Store output = context.reduction(0);
  auto sum             = output.reduce_accessor<Reduce, true, 1>();
  // Best-practice is to validate types
  assert(output.code() == legate::Type::Code::FLOAT32);
  assert(output.dim() == 1);
  assert(output.shape<1>() == legate::Rect<1>(0, 0));
  sum.reduce(0, total);
}

/*static*/ void SquareTask::cpu_variant(legate::TaskContext context)
{
  legate::Store output = context.output(0);
  // Best-practice to validate the store types
  assert(output.code() == legate::Type::Code::FLOAT32);
  assert(output.dim() == 1);
  legate::Rect<1> output_shape = output.shape<1>();
  auto out                     = output.write_accessor<float, 1>();

  legate::Store input = context.input(0);
  // Best-practice to validate the store types
  assert(input.code() == legate::Type::Code::FLOAT32);
  assert(input.dim() == 1);
  legate::Rect<1> input_shape = input.shape<1>();  // should be a 1-Dim array
  auto in                     = input.read_accessor<float, 1>();

  assert(input_shape == output_shape);

  logger.info() << "Elementwise square [" << output_shape.lo << "," << output_shape.hi << "]";

  // i is a global index for the complete array
  for (int64_t i = input_shape.lo; i <= input_shape.hi; ++i) { out[i] = in[i] * in[i]; }
}

/*static*/ void IotaTask::cpu_variant(legate::TaskContext context)
{
  legate::Store output         = context.output(0);
  legate::Rect<1> output_shape = output.shape<1>();
  auto out                     = output.write_accessor<float, 1>();

  logger.info() << "Iota task [" << output_shape.lo << "," << output_shape.hi << "]";

  // i is a global index for the complete array
  for (int64_t i = output_shape.lo; i <= output_shape.hi; ++i) { out[i] = i + 1; }
}

}  // namespace hello

}  // namespace task
