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

#include "core/legate_c.h"
#include "core/task/task_context.h"

namespace legate::detail {

/*static*/ void FixupRanges::cpu_variant(legate::TaskContext context)
{
  if (context.get_task_index()[0] == 0) {
    return;
  }
  // TODO(wonchanl): We need to extend this to nested cases
  for (auto&& output : context.outputs()) {
    auto list_arr = output.as_list_array();

    auto list_desc  = list_arr.descriptor();
    auto desc_shape = list_desc.shape<1>();
    if (desc_shape.empty()) {
      continue;
    }

    const auto vardata_lo = list_arr.vardata().shape<1>().lo;
    auto desc_acc         = list_desc.data().read_write_accessor<Rect<1>, 1>();

    for (auto idx = desc_shape.lo[0]; idx <= desc_shape.hi[0]; ++idx) {
      auto& desc = desc_acc[idx];
      desc.lo += vardata_lo;
      desc.hi += vardata_lo;
    }
  }
}

/*static*/ void OffsetsToRanges::cpu_variant(legate::TaskContext context)
{
  const auto offsets = context.input(0).data();
  const auto vardata = context.input(1).data();
  const auto ranges  = context.output(0).data();

  const auto shape = offsets.shape<1>();
  LegateCheck(shape == ranges.shape<1>());

  if (shape.empty()) {
    return;
  }

  const auto vardata_shape = vardata.shape<1>();
  const auto vardata_lo    = vardata_shape.lo[0];

  const auto offsets_acc = offsets.read_accessor<int32_t, 1>();
  const auto ranges_acc  = ranges.write_accessor<Rect<1>, 1>();
  for (auto idx = shape.lo[0]; idx < shape.hi[0]; ++idx) {
    ranges_acc[idx].lo[0] = vardata_lo + offsets_acc[idx];
    ranges_acc[idx].hi[0] = vardata_lo + offsets_acc[idx + 1] - 1;
  }
  ranges_acc[shape.hi].lo[0] = vardata_lo + offsets_acc[shape.hi];
  ranges_acc[shape.hi].hi[0] = vardata_lo + static_cast<std::int64_t>(vardata_shape.volume()) - 1;
}

/*static*/ void RangesToOffsets::cpu_variant(legate::TaskContext context)
{
  const auto ranges  = context.input(0).data();
  const auto offsets = context.output(0).data();

  const auto shape = ranges.shape<1>();
  LegateCheck(shape == offsets.shape<1>());

  if (shape.empty()) {
    return;
  }

  const auto ranges_acc  = ranges.read_accessor<Rect<1>, 1>();
  const auto offsets_acc = offsets.write_accessor<int32_t, 1>();
  const auto lo          = ranges_acc[shape.lo].lo[0];
  for (auto idx = shape.lo[0]; idx <= shape.hi[0]; ++idx) {
    offsets_acc[idx] = static_cast<std::int32_t>(ranges_acc[idx].lo[0] - lo);
  }
}

void register_array_tasks(Library* core_lib)
{
  FixupRanges::register_variants(legate::Library(core_lib), LEGATE_CORE_FIXUP_RANGES);
  OffsetsToRanges::register_variants(legate::Library(core_lib), LEGATE_CORE_OFFSETS_TO_RANGES);
  RangesToOffsets::register_variants(legate::Library(core_lib), LEGATE_CORE_RANGES_TO_OFFSETS);
}

}  // namespace legate::detail
