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

#include "task_simple.h"

namespace task::simple {

// NOLINTBEGIN(readability-magic-numbers)

void register_tasks()
{
  static bool prepared = false;
  if (prepared) {
    return;
  }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  HelloTask::register_variants(context);
  WriterTask::register_variants(context);
  ReducerTask::register_variants(context);
}

/*static*/ void HelloTask::cpu_variant(legate::TaskContext context)
{
  auto output = context.output(0).data();
  auto shape  = output.shape<2>();

  if (shape.empty()) {
    return;
  }

  auto acc = output.write_accessor<int64_t, 2>(shape);
  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    acc[*it] = (*it)[0] + (*it)[1] * 1000;
  }
}

/*static*/ void WriterTask::cpu_variant(legate::TaskContext context)
{
  auto output1 = context.output(0).data();
  auto output2 = context.output(1).data();

  auto acc1 = output1.write_accessor<int32_t, 2>();
  auto acc2 = output2.write_accessor<int64_t, 3>();

  acc1[{0, 0}]    = 10;
  acc2[{0, 0, 0}] = 20;
}

/*static*/ void ReducerTask::cpu_variant(legate::TaskContext context)
{
  auto in   = context.input(0).data();
  auto red1 = context.reduction(0).data();
  auto red2 = context.reduction(1).data();

  auto shape = in.shape<1>();
  if (shape.empty()) {
    return;
  }

  auto red_acc1 = red1.reduce_accessor<legate::SumReduction<std::int32_t>, true, 2>();
  auto red_acc2 = red2.reduce_accessor<legate::ProdReduction<std::int64_t>, true, 3>();

  for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
    red_acc1[{0, 0}].reduce(10);
    red_acc2[{0, 0, 0}].reduce(2);
  }
}

// NOLINTEND(readability-magic-numbers)

}  // namespace task::simple
