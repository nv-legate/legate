/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/ompthreads.h>

#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/runtime/detail/argument_parsing/exceptions.h>
#include <legate/runtime/detail/config.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/traced_exception.h>

#include <realm/module_config.h>

#include <fmt/core.h>

#include <cmath>
#include <cstdint>

namespace legate::detail {

namespace {

void configure_ompthreads_impl(bool auto_config,
                               const Realm::ModuleConfig& core,
                               const Argument<std::int32_t>& util,
                               const Argument<std::int32_t>& cpus,
                               const Argument<std::int32_t>& gpus,
                               const Argument<std::int32_t>& omps,
                               Argument<std::int32_t>* ompthreads)
{
  LEGATE_CHECK(util.value() >= 0);
  LEGATE_CHECK(cpus.value() >= 0);
  LEGATE_CHECK(gpus.value() >= 0);
  LEGATE_CHECK(omps.value() >= 0);

  if (ompthreads->value() >= 0) {
    return;
  }

  if (omps.value() == 0) {
    ompthreads->value_mut() = 0;
    return;
  }

  if (!auto_config) {
    ompthreads->value_mut() = 1;
    return;
  }

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
      "Core Realm module detected 0 CPU cores while configuring the number of OpenMP threads."};
  }

  const auto auto_ompthreads = static_cast<std::int32_t>(
    std::floor((res_num_cpus - cpus.value() - util.value() - gpus.value()) / omps.value()));

  if (auto_ompthreads <= 0) {
    throw TracedException<AutoConfigurationError>{
      fmt::format("Not enough CPU cores to split across {} OpenMP processor(s). Have {}, but need "
                  "{} for CPU processors, {} for utility processors, {} for GPU processors, and at "
                  "least {} for OpenMP processors (1 core each).",
                  omps.value(),
                  res_num_cpus,
                  cpus.value(),
                  util.value(),
                  gpus.value(),
                  omps.value())};
  }
  ompthreads->value_mut() = auto_ompthreads;
}

}  // namespace

void configure_ompthreads(bool auto_config,
                          const Realm::ModuleConfig& core,
                          const Argument<std::int32_t>& util,
                          const Argument<std::int32_t>& cpus,
                          const Argument<std::int32_t>& gpus,
                          const Argument<std::int32_t>& omps,
                          Argument<std::int32_t>* ompthreads,
                          Config* cfg)
{
  configure_ompthreads_impl(auto_config, core, util, cpus, gpus, omps, ompthreads);
  if (omps.value() > 0) {
    const auto num_threads = ompthreads->value();

    LEGATE_CHECK(num_threads > 0);
    cfg->set_need_openmp(true);
    cfg->set_num_omp_threads(num_threads);
  }
}

}  // namespace legate::detail
