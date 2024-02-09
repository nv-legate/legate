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

#include "hello_world.h"
#include "legate_library.h"

namespace hello {

class SumTask : public Task<SumTask, SUM> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    legate::Store input         = context.input(0).data();
    legate::Rect<1> input_shape = input.shape<1>();  // should be a 1-Dim array
    auto in                     = input.read_accessor<float, 1>();

    logger.info() << "Sum [" << input_shape.lo << "," << input_shape.hi << "]";

    float total = 0;
    // i is a global index for the complete array
    for (size_t i = input_shape.lo; i <= input_shape.hi; ++i) {
      total += in[i];
    }

    /**
      The task launch as a whole will return a single value (Store of size 1)
      to the caller. However, each point task gets a separate Store of the
      same size as the result, to reduce into. This "local accumulator" will
      be initialized by the runtime, and all we need to do is call .reduce()
      to add our local contribution. After all point tasks return, the runtime
      will make sure to combine all their buffers into the single final result.
    */
    using Reduce         = Legion::SumReduction<float>;
    legate::Store output = context.reduction(0).data();
    auto sum             = output.reduce_accessor<Reduce, true, 1>();
    LegateCheck(output.shape<1>() == legate::Rect<1>(0, 0));
    sum.reduce(0, total);
  }
};

}  // namespace hello

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  hello::SumTask::register_variants();
}

}  // namespace
