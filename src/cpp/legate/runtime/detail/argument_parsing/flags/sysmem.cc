/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/sysmem.h>

#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/runtime/detail/argument_parsing/exceptions.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/traced_exception.h>

#include <realm/module_config.h>

#include <fmt/core.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace legate::detail {

void configure_sysmem(bool auto_config,
                      const Realm::ModuleConfig& core,
                      const Argument<Scaled<std::int64_t>>& numamem,
                      Argument<Scaled<std::int64_t>>* sysmem)
{
  auto& sysmem_value = sysmem->value_mut().unscaled_value_mut();

  if (sysmem_value >= 0) {
    return;
  }

  constexpr auto MINIMAL_MEM = 256;

  if (!auto_config) {
    sysmem_value = MINIMAL_MEM;
    return;
  }

  if (numamem.value().scaled_value() > 0) {
    // don't allocate much memory to --sysmem; leave most to be used for --numamem
    sysmem_value = MINIMAL_MEM;
    return;
  }

  std::size_t res_sysmem_size{};

  if (auto realm_status = core.get_resource("sysmem", res_sysmem_size);
      realm_status != REALM_SUCCESS) {
    // system memory must be available
    LEGATE_CHECK(realm_status != REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
    if (realm_status == REALM_MODULE_CONFIG_ERROR_NO_RESOURCE) {
      throw TracedException<AutoConfigurationError>{
        "Core Realm module could not determine the available system memory."};
    }
    throw TracedException<AutoConfigurationError>{
      fmt::format("Core Realm module encountered an unknown error while determining the available "
                  "system memory, error {}.",
                  static_cast<int>(realm_status))};
  }

  constexpr double SYSMEM_FRACTION = 0.8;
  const auto alloc_frac            = SYSMEM_FRACTION * static_cast<double>(res_sysmem_size);
  const auto scale                 = static_cast<double>(sysmem->value().scale());
  const auto auto_sysmem           = std::floor(alloc_frac / scale);

  sysmem_value = static_cast<std::int64_t>(auto_sysmem);
}

}  // namespace legate::detail
