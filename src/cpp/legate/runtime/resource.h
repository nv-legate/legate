/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>

#include <cstdint>

namespace legate {

/**
 * @addtogroup runtime
 * @{
 */

/**
 * @brief POD for library configuration.
 */
struct ResourceConfig {
  /**
   * @brief Maximum number of tasks that the library can register
   */
  std::int64_t max_tasks{1024};  // NOLINT(readability-magic-numbers)
  /**
   * @brief Maximum number of dynamic tasks that the library can register (cannot exceed max_tasks)
   */
  std::int64_t max_dyn_tasks{0};
  /**
   * @brief Maximum number of custom reduction operators that the library can register
   */
  std::int64_t max_reduction_ops{};
  std::int64_t max_projections{};
  std::int64_t max_shardings{};
};

/** @} */

}  // namespace legate
