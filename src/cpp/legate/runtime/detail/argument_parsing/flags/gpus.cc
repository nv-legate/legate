/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/gpus.h>

#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/runtime/detail/argument_parsing/exceptions.h>
#include <legate/runtime/detail/config.h>
#include <legate/utilities/detail/traced_exception.h>

#include <realm/module_config.h>

#include <fmt/core.h>

namespace legate::detail {

namespace {

void configure_gpus_impl(bool auto_config,
                         const Realm::ModuleConfig* cuda,
                         Argument<std::int32_t>* gpus)
{
  if (gpus->value() >= 0) {
    return;
  }

  if (auto_config && (cuda != nullptr)) {
    std::int32_t auto_gpus = 0;

    // use all available GPUs
    if (auto realm_status = cuda->get_resource("gpu", auto_gpus); realm_status != REALM_SUCCESS) {
      if (realm_status == REALM_MODULE_CONFIG_ERROR_INVALID_NAME) {
        throw TracedException<AutoConfigurationError>{
          "CUDA Realm module is not configured for GPUs."};
      }
      if (realm_status == REALM_MODULE_CONFIG_ERROR_NO_RESOURCE) {
        throw TracedException<AutoConfigurationError>{
          "CUDA Realm module could not determine the number of GPUs."};
      }
      throw TracedException<AutoConfigurationError>{
        fmt::format("CUDA Realm module encountered an unknown error while determining the number "
                    "of GPUs, error {}.",
                    static_cast<int>(realm_status))};
    }
    gpus->value_mut() = auto_gpus;
  } else {
    // otherwise don't allocate any GPUs
    gpus->value_mut() = 0;
  }
}

}  // namespace

void configure_gpus(bool auto_config,
                    const Realm::ModuleConfig* cuda,
                    Argument<std::int32_t>* gpus,
                    Config* cfg)
{
  configure_gpus_impl(auto_config, cuda, gpus);
  if (gpus->value() > 0) {
    cfg->set_need_cuda(true);
  }
}

}  // namespace legate::detail
