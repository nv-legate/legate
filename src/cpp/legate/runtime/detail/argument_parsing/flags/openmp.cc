/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/openmp.h>

#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/utilities/span.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace legate::detail {

void configure_omps(bool auto_config,
                    const Realm::ModuleConfig* openmp,
                    Span<const std::size_t> numa_mems,
                    const Argument<std::int32_t>& gpus,
                    Argument<std::int32_t>* omps)
{
  if (omps->value() >= 0) {
    return;
  }

  if (!auto_config || !openmp) {
    omps->value_mut() = 0;  // don't allocate any OpenMP groups
    return;
  }

  if (gpus.value() > 0) {
    // match the number of GPUs, to ensure host offloading does not repartition
    omps->value_mut() = gpus.value();
    return;
  }

  // create one OpenMP group per NUMA node (or a single group, if no NUMA info is available)
  omps->value_mut() = std::max(static_cast<std::int32_t>(numa_mems.size()), 1);
}

}  // namespace legate::detail
