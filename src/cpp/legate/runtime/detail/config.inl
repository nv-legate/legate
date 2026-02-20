/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/detail/config.h>

#include <realm/module_config.h>
#include <realm/runtime.h>

namespace legate::detail {

template <typename T>
std::optional<T> get_realm_config_property(const std::string& module_name,
                                           const std::string& property_name)
{
  auto* const config = Realm::Runtime::get_runtime().get_module_config(module_name);

  if (config == nullptr) {
    return {};
  }

  T value{};

  if (auto realm_status = config->get_property(property_name, value);
      realm_status == REALM_MODULE_CONFIG_ERROR_INVALID_NAME) {
    return {};
  }

  return value;
}

}  // namespace legate::detail
