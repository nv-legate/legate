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

class IotaTask : public Task<IotaTask, IOTA> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    legate::Store output         = context.output(0).data();
    legate::Rect<1> output_shape = output.shape<1>();
    auto out                     = output.write_accessor<float, 1>();

    logger.info() << "Iota task [" << output_shape.lo << "," << output_shape.hi << "]";

    // i is a global index for the complete array
    for (size_t i = output_shape.lo; i <= output_shape.hi; ++i) { out[i] = i + 1; }
  }
};

}  // namespace hello

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks(void)
{
  hello::IotaTask::register_variants();
}

}  // namespace
