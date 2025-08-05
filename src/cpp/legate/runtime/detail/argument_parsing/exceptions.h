/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <stdexcept>
#include <string_view>

namespace legate::detail {

/**
 * @brief Exception thrown during Legate startup when configuration fails.
 *
 * This exception implies that the Legate runtime failed to start. The error behind this exception
 * is most likely not recoverable, and restarting the Legate runtime in the same process will likely
 * fail.
 *
 * The underlying issue is likely that the caller requested a resource that does not exist on the
 * current machine, or is not supported by the current build of Legate (e.g. requested GPUs in a
 * CPU-only build of Legate). The caller should adjust the options specified in ``LEGATE_CONFIG``
 * before restarting the application and calling ``legate::start`` again.
 */
class LEGATE_EXPORT ConfigurationError : public std::runtime_error {
 public:
  /**
   * @brief Create a `ConfigurationError` with the given explanatory message.
   *
   * @param msg The explanatory message
   */
  explicit ConfigurationError(std::string_view msg);
};

/**
 * @brief Exception thrown during Legate startup when the automatic configuration heuristics fail.
 *
 * This exception implies that the Legate runtime failed to start. The error behind this exception
 * is most likely not recoverable, and restarting the Legate runtime in the same process will likely
 * fail.
 *
 * The underlying issue is that Legate was unable to synthesize a suitable configuration, either
 * because hardware detection failed, or the detected resources were not enough to compute a sane
 * configuration. The caller should manually specify the configuration using ``LEGATE_CONFIG``,
 * and/or disable automatic configuration altogether with ``LEGATE_AUTO_CONFIG=0``, before
 * restarting the application and calling ``legate::start`` again.
 */
class LEGATE_EXPORT AutoConfigurationError : public std::runtime_error {
 public:
  /**
   * @brief Create an `AutoConfigurationError` with the given explanatory message.
   *
   * @param msg The explanatory message
   */
  explicit AutoConfigurationError(std::string_view msg);
};

}  // namespace legate::detail
