/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/config_realm.h>

#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/runtime/detail/argument_parsing/exceptions.h>
#include <legate/runtime/detail/argument_parsing/parse.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/traced_exception.h>

#include <realm/module_config.h>
#include <realm/runtime.h>

#include <fmt/core.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace legate::detail {

namespace {

template <typename T>
[[nodiscard]] T arg_value(const Argument<T>& arg)
{
  return arg.value();
}

template <typename T>
[[nodiscard]] T arg_value(const Argument<Scaled<T>>& arg)
{
  return arg.value().scaled_value();
}

template <typename T>
void try_set_property(const std::string& module_name,
                      const std::string& property_name,
                      const Argument<T>& arg)
{
  auto* const config = Realm::Runtime::get_runtime().get_module_config(module_name);

  if (nullptr == config) {
    // If the module doesn't exist, but the user never set the value, we don't care
    if (!arg.was_set()) {
      return;
    }

    throw TracedException<ConfigurationError>{fmt::format(
      "Unable to set {}->{} from flag {} (the Realm {} module is not available). This may indicate "
      "that Legate is trying to configure a property for which it was not configured for (for "
      "example, setting CUDA properties without CUDA support). Or, if Legate does have support for "
      "this feature, it may indicate that your system does not support the feature (for example, "
      "setting CUDA properties on a system which has CUDA libraries, but no enabled GPUs). "
      "Finally, it may also indicate that Realm previously failed to initialize the {} module "
      "(unlikely).",
      module_name,
      property_name,
      arg.flag(),
      module_name,
      module_name)};
  }

  const auto value = arg_value(arg);

  LEGATE_CHECK(value >= 0);
  if (auto realm_status = config->set_property(property_name, value);
      realm_status != REALM_SUCCESS) {
    const auto error_detail = [&]() -> std::string {
      if (realm_status == REALM_MODULE_CONFIG_ERROR_INVALID_NAME) {
        return "Module doesn't have the configuration.";
      }
      return fmt::format("Unknown error: {}.", static_cast<int>(realm_status));
    }();

    throw TracedException<ConfigurationError>{
      fmt::format("Realm failed to set module configuration. Module: {} Configuration: {} Value: "
                  "{} (from flag {}). {}",
                  module_name,
                  property_name,
                  value,
                  arg.flag(),
                  error_detail)};
  }
}

void set_core_config_properties(const Argument<std::int32_t>& cpus,
                                const Argument<std::int32_t>& util,
                                const Argument<Scaled<std::int64_t>>& sysmem,
                                const Argument<Scaled<std::int64_t>>& regmem)
{
  try_set_property("core", "cpu", cpus);
  try_set_property("core", "util", util);
  try_set_property("core", "sysmem", sysmem);
  try_set_property("core", "regmem", regmem);
  // Don't register sysmem for intra-node IPC if it's above a certain size, as it can take
  // forever.
  const std::size_t SYSMEM_LIMIT_FOR_IPC_REG =
    1024 * static_cast<std::size_t>(sysmem.value().scale());
  const auto core_mod = Realm::Runtime::get_runtime().get_module_config("core");

  LEGATE_CHECK(core_mod != nullptr);
  static_cast<void>(core_mod->set_property("sysmem_ipc_limit", SYSMEM_LIMIT_FOR_IPC_REG));
}

void set_cuda_config_properties(const Argument<std::int32_t>& gpus,
                                const Argument<Scaled<std::int64_t>>& fbmem,
                                const Argument<Scaled<std::int64_t>>& zcmem)
{
  try {
    try_set_property("cuda", "gpu", gpus);
    try_set_property("cuda", "fbmem", fbmem);
    try_set_property("cuda", "zcmem", zcmem);
  } catch (...) {
    // If we have CUDA, but failed above, then rethrow, otherwise silently gobble the error
    if (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
      throw;
    }
  }
}

void set_openmp_config_properties(const Argument<std::int32_t>& omps,
                                  const Argument<std::int32_t>& ompthreads,
                                  const Argument<Scaled<std::int64_t>>& numamem)
{
  try {
    try_set_property("openmp", "ocpu", omps);
    try_set_property("openmp", "othr", ompthreads);
    try_set_property("numa", "numamem", numamem);
  } catch (...) {
    // If we have OpenMP, but failed above, then rethrow, otherwise silently gobble the error
    if (LEGATE_DEFINED(LEGATE_USE_OPENMP)) {
      throw;
    }
  }
}

}  // namespace

void configure_realm(const ParsedArgs& parsed)
{
  set_core_config_properties(parsed.cpus, parsed.util, parsed.sysmem, parsed.regmem);
  set_cuda_config_properties(parsed.gpus, parsed.fbmem, parsed.zcmem);
  set_openmp_config_properties(parsed.omps, parsed.ompthreads, parsed.numamem);
}

}  // namespace legate::detail
