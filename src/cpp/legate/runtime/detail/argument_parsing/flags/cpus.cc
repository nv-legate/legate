/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/cpus.h>

#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/runtime/detail/argument_parsing/exceptions.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/traced_exception.h>

#include <realm/module_config.h>

#include <fmt/core.h>

#include <cstdint>

namespace legate::detail {

void configure_cpus(bool auto_config,
                    const Realm::ModuleConfig& core,
                    const Argument<std::int32_t>& omps,
                    const Argument<std::int32_t>& util,
                    const Argument<std::int32_t>& gpus,
                    Argument<std::int32_t>* cpus)
{
  auto& cpus_value = cpus->value_mut();

  // If any of these are negative, it means we forgot to configure them first.
  LEGATE_CHECK(omps.value() >= 0);
  LEGATE_CHECK(gpus.value() >= 0);
  LEGATE_CHECK(util.value() >= 0);
  if (cpus_value >= 0) {
    return;
  }

  if (!auto_config || (omps.value() > 0)) {
    // leave one core available for profiling meta-tasks, and other random uses
    cpus_value = 1;
    return;
  }

  if (gpus.value() > 0) {
    // match the number of GPUs, to ensure host offloading does not repartition
    cpus_value = gpus.value();
    return;
  }

  // use all unallocated cores
  int res_num_cpus{};

  if (auto realm_status = core.get_resource("cpu", res_num_cpus); realm_status != REALM_SUCCESS) {
    // cpu cores must be available
    LEGATE_CHECK(realm_status != REALM_MODULE_CONFIG_ERROR_INVALID_NAME);
    if (realm_status == REALM_MODULE_CONFIG_ERROR_NO_RESOURCE) {
      throw TracedException<AutoConfigurationError>{
        "Core Realm module could not determine the number of CPU cores."};
    }
    throw TracedException<AutoConfigurationError>{
      fmt::format("Core Realm module encountered an unknown error while determining the number of "
                  "CPU cores, error {}.",
                  static_cast<int>(realm_status))};
  }
  if (res_num_cpus == 0) {
    throw TracedException<AutoConfigurationError>{
      "Core Realm module detected 0 CPU cores while configuring CPUs."};
  }

  // Technically, can remove gpus.value() here, due to the above early-exit we know it is
  // always 0 at this point
  const auto auto_cpus = res_num_cpus - util.value() - gpus.value();

  if (auto_cpus <= 0) {
    throw TracedException<AutoConfigurationError>{
      fmt::format("No CPU cores left to allocate to CPU processors. Have {}, but need {} for "
                  "utility processors, and {} for GPU processors.",
                  res_num_cpus,
                  util.value(),
                  gpus.value())};
  }

  cpus_value = auto_cpus;
}

}  // namespace legate::detail
