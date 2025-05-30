/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/array_tasks.h>

#include <legate/task/task_context.h>

namespace legate::detail {

/*static*/ void FixupRanges::cpu_variant(legate::TaskContext context)
{
  if (context.get_task_index()[0] == 0) {
    return;
  }
  // TODO(wonchanl): We need to extend this to nested cases
  const auto num_outputs = context.num_outputs();

  for (std::uint32_t i = 0; i < num_outputs; ++i) {
    const auto list_arr   = context.output(i).as_list_array();
    const auto vardata_lo = list_arr.vardata().shape<1>().lo;
    const auto desc_acc   = list_arr.descriptor().data().span_read_write_accessor<Rect<1>, 1>();

    for (coord_t idx = 0; idx < desc_acc.extent(0); ++idx) {
      auto& v = desc_acc(idx);

      v.lo += vardata_lo;
      v.hi += vardata_lo;
    }
  }
}

void register_array_tasks(Library& core_lib)
{
  FixupRanges::register_variants(legate::Library{&core_lib});
}

}  // namespace legate::detail
