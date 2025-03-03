/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/array_tasks.h>
#include <legate/task/task_context.h>

namespace legate::detail {

/*static*/ void FixupRanges::omp_variant(legate::TaskContext context)
{
  if (context.get_task_index()[0] == 0) {
    return;
  }

  // TODO(wonchanl): We need to extend this to nested cases
  const auto num_outputs = context.num_outputs();

  for (std::uint32_t i = 0; i < num_outputs; ++i) {
    const auto list_arr   = context.output(i).as_list_array();
    const auto list_desc  = list_arr.descriptor();
    const auto desc_shape = list_desc.shape<1>();

    if (desc_shape.empty()) {
      continue;
    }

    const auto vardata_lo = list_arr.vardata().shape<1>().lo;
    const auto desc_acc   = list_desc.data().read_write_accessor<Rect<1>, 1>();

#pragma omp parallel for schedule(static)
    for (std::int64_t idx = desc_shape.lo[0]; idx <= desc_shape.hi[0]; ++idx) {
      auto& desc = desc_acc[idx];
      desc.lo += vardata_lo;
      desc.hi += vardata_lo;
    }
  }
}

}  // namespace legate::detail
