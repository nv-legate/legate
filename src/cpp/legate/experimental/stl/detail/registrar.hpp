/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

#include <legate/experimental/stl/detail/config.hpp>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

namespace legate::experimental::stl {

////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Configuration for the Legate STL resource.
 *
 * This constant represents the configuration for the Legate STL resource. It specifies (in order):
 * @li the maximum number of tasks,
 * @li the maximum number of dynamic tasks,
 * @li the maximum number of reduction operations,
 * @li the maximum number of projections, and
 * @li the maximum number of shardings
 *
 * that can be used in a program using Legate.STL.
 *
 * @see \c initialize_library
 * @ingroup stl-utilities
 */
inline constexpr ResourceConfig LEGATE_STL_RESOURCE_CONFIG = {
  1024,  //< max_tasks{1024};
  1024,  //< max_dyn_tasks{0};
  64,    //< max_reduction_ops{};
  0,     //< max_projections{};
  0      //< max_shardings{};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * @class initialize_library
 * @brief A class that initializes the Legate runtime and creates the
 * `legate.stl` library instance.
 *
 * The `initialize_library` class is responsible for creating a library instance. The initialization
 * fails if the runtime has not started. If the initialization is successful, it creates a library
 * with the name @c "legate.stl".
 *
 * The library instance is automatically destroyed when the `initialize_library` object goes out of
 * scope.
 *
 * It is harmless to create multiple `initialize_library` objects in the same program.
 *
 * @see @c LEGATE_STL_RESOURCE_CONFIG
 * @ingroup stl-utilities
 */
class initialize_library {  // NOLINT(readability-identifier-naming)
 public:
  /**
   * @brief Constructs an @c initialize_library object.
   *
   * This constructor creates a library instance for Legate STL. The initialization fails if the
   * runtime has not started. If the initialization is successful, it creates a library with the
   * name @c "legate.stl".
   *
   * @throw std::runtime_error If the runtime has not started
   */
  initialize_library()
    : library_{legate::has_started()
                 ? legate::Runtime::get_runtime()->find_or_create_library(
                     "legate.stl", LEGATE_STL_RESOURCE_CONFIG)
                 : throw legate::detail::TracedException<std::runtime_error>{
                     "Legate STL requires the Legate runtime to be started first"}}
  {
  }

 private:
  Library library_;  ///< The library instance.
};

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
