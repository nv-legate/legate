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

#include "core/data/detail/array_tasks.h"
#include "core/task/task_context.h"

namespace legate::detail {

/*static*/ void FixupRanges::omp_variant(legate::TaskContext context)
{
  if (context.get_task_index()[0] == 0) {
    return;
  }

  auto outputs = context.outputs();
  // TODO(wonchanl): We need to extend this to nested cases
  for (auto& output : outputs) {
    auto list_arr = output.as_list_array();

    auto list_desc  = list_arr.descriptor();
    auto desc_shape = list_desc.shape<1>();
    if (desc_shape.empty()) {
      continue;
    }

    auto vardata_lo = list_arr.vardata().shape<1>().lo;
    auto desc_acc   = list_desc.data().read_write_accessor<Rect<1>, 1>();

#pragma omp parallel for schedule(static)
    for (int64_t idx = desc_shape.lo[0]; idx <= desc_shape.hi[0]; ++idx) {
      auto& desc = desc_acc[idx];
      desc.lo += vardata_lo;
      desc.hi += vardata_lo;
    }
  }
}

/*static*/ void OffsetsToRanges::omp_variant(legate::TaskContext context)
{
  auto offsets = context.input(0).data();
  auto vardata = context.input(1).data();
  auto ranges  = context.output(0).data();

  auto shape = offsets.shape<1>();
  LegateCheck(shape == ranges.shape<1>());

  if (shape.empty()) {
    return;
  }

  auto vardata_shape = vardata.shape<1>();
  auto vardata_lo    = vardata_shape.lo[0];

  auto offsets_acc = offsets.read_accessor<int32_t, 1>();
  auto ranges_acc  = ranges.write_accessor<Rect<1>, 1>();
#pragma omp parallel for schedule(static)
  for (int64_t idx = shape.lo[0]; idx < shape.hi[0]; ++idx) {
    ranges_acc[idx].lo[0] = vardata_lo + offsets_acc[idx];
    ranges_acc[idx].hi[0] = vardata_lo + offsets_acc[idx + 1] - 1;
  }
  ranges_acc[shape.hi].lo[0] = vardata_lo + offsets_acc[shape.hi];
  ranges_acc[shape.hi].hi[0] = vardata_lo + static_cast<int64_t>(vardata_shape.volume()) - 1;
}

/*static*/ void RangesToOffsets::omp_variant(legate::TaskContext context)
{
  auto ranges  = context.input(0).data();
  auto offsets = context.output(0).data();

  auto shape = ranges.shape<1>();
  LegateCheck(shape == offsets.shape<1>());

  if (shape.empty()) {
    return;
  }

  auto ranges_acc  = ranges.read_accessor<Rect<1>, 1>();
  auto offsets_acc = offsets.write_accessor<int32_t, 1>();
  auto lo          = ranges_acc[shape.lo].lo[0];
#pragma omp parallel for schedule(static)
  for (int64_t idx = shape.lo[0]; idx <= shape.hi[0]; ++idx) {
    offsets_acc[idx] = static_cast<int32_t>(ranges_acc[idx].lo[0] - lo);
  }
}

}  // namespace legate::detail
