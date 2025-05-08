/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/numamem.h>

#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/utilities/span.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace legate::detail {

void configure_numamem(bool auto_config,
                       Span<const std::size_t> numa_mems,
                       const Argument<std::int32_t>& omps,
                       Argument<Scaled<std::int64_t>>* numamem)
{
  auto& numamem_value = numamem->value_mut().unscaled_value_mut();

  if (numamem_value >= 0) {
    return;
  }

  // Negative value here indicates we forgot to configure openmp before calling this function.
  LEGATE_CHECK(omps.value() >= 0);
  if (omps.value() == 0 || numa_mems.size() == 0 || omps.value() % numa_mems.size() != 0) {
    numamem_value = 0;
    return;
  }

  if (!auto_config) {
    constexpr auto MINIMAL_MEM = 256;

    numamem_value = MINIMAL_MEM;
    return;
  }

  // TODO(mpapadakis): Assuming that all NUMA domains have the same size
  constexpr double SYSMEM_FRACTION = 0.8;
  const auto numa_mem_size         = numa_mems.front();
  const auto num_numa_mems         = numa_mems.size();
  const auto omps_per_numa         = (omps.value() + num_numa_mems - 1) / num_numa_mems;
  // Evenly divide 80% of the available NUMA memory across all OpenMP threads
  const auto alloc_frac = SYSMEM_FRACTION * static_cast<double>(numa_mem_size);
  const auto scale      = static_cast<double>(numamem->value().scale());
  const auto auto_numamem =
    static_cast<std::int64_t>(std::floor(alloc_frac / scale / static_cast<double>(omps_per_numa)));

  numamem_value = auto_numamem;
}

}  // namespace legate::detail
