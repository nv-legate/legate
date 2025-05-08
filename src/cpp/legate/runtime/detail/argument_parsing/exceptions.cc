/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/exceptions.h>

#include <fmt/core.h>

namespace legate::detail {

ConfigurationError::ConfigurationError(std::string_view msg)
  : runtime_error{fmt::format(
      "Legate configuration failed: {} Make sure the selected options (inspect with "
      "LEGATE_SHOW_CONFIG=1) are appropriate for the current machine and build of Legate.",
      msg)}
{
}

AutoConfigurationError::AutoConfigurationError(std::string_view msg)
  : runtime_error{fmt::format(
      "Legate auto-configuration failed: {} Use LEGATE_CONFIG to set configuration parameters "
      "manually, and/or disable automatic configuration with LEGATE_AUTO_CONFIG=0.",
      msg)}
{
}

}  // namespace legate::detail
