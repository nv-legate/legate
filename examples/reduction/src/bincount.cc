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

#include "core/utilities/dispatch.h"

#include "legate_library.h"
#include "reduction_cffi.h"

namespace reduction {

class BincountTask : public Task<BincountTask, BINCOUNT> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0).data();
    auto output = context.reduction(0).data();

    auto in_shape  = input.shape<1>();
    auto out_shape = output.shape<1>();

    auto in_acc  = input.read_accessor<uint64_t, 1>();
    auto out_acc = output.reduce_accessor<legate::SumReduction<std::uint64_t>, true, 1>();

    for (legate::PointInRectIterator<1> it(in_shape); it.valid(); ++it) {
      auto& value = in_acc[*it];
      legate::Point<1> pos_reduce(static_cast<std::int64_t>(value));

      if (out_shape.contains(pos_reduce)) {
        out_acc.reduce(pos_reduce, 1);
      }
    }
  }
};

}  // namespace reduction

namespace {

static void __attribute__((constructor)) register_tasks()
{
  reduction::BincountTask::register_variants();
}

}  // namespace
