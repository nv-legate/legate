/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "config.hpp"
#include "legate.h"

// Include this last:
#include "prefix.hpp"

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
 * The `initialize_library` class is responsible for initializing the Legate
 * runtime and creating a library instance. It takes the program's command line
 * arguments and initializes the Legate runtime with them. If the initialization
 * is successful, it creates a library with the name `legate.stl` using the
 * `LEGATE_STL_RESOURCE_CONFIG` configuration.
 *
 * The library instance is automatically destroyed when the `initialize_library`
 * object goes out of scope.
 *
 * It is harmless to create multiple `initialize_library` objects in the same
 * program. The Legate runtime is initialized only once and only one instance of
 * the `legate.stl` library will be created; however, the destruction of any
 * `initialize_library` object finalizes the Legate runtime.
 *
 * @see @c LEGATE_STL_RESOURCE_CONFIG
 * @ingroup stl-utilities
 */
class initialize_library {  // NOLINT(readability-identifier-naming)
 public:
  /**
   * @brief Constructs an @c initialize_library object.
   *
   * This constructor initializes the Legate library and creates a library instance.
   * It takes the command line arguments and starts the Legate runtime. If the
   * initialization is successful, it creates a library with the name @c "legate.stl"
   * using the @c LEGATE_STL_RESOURCE_CONFIG configuration.
   *
   * @param argc The number of command line arguments.
   * @param argv An array of C-style strings representing the command line arguments.
   */
  initialize_library(int argc, char* argv[])
    : result_{legate::start(argc, argv)},
      library_{result() == 0 ? legate::Runtime::get_runtime()->find_or_create_library(
                                 "legate.stl", LEGATE_STL_RESOURCE_CONFIG)
                             : Library{nullptr}}
  {
  }

  /**
   * @brief Destroys the @c initialize_library object.
   *
   * This destructor finalizes the Legate library if the initialization was successful.
   */
  ~initialize_library()
  {
    if (result() == 0) {
      result_ = legate::finish();
    }
  }

  /**
   * @brief Get the result of the library initialization.
   *
   * This function returns the result of the library initialization. If the result is
   * zero, the initialization was successful. Otherwise, it indicates an error code.
   *
   * @return The result of the library initialization.
   */
  [[nodiscard]] std::int32_t result() const { return result_; }

 private:
  std::int32_t result_{};  ///< The result of the library initialization.
  Library library_;        ///< The library instance.
};

}  // namespace legate::experimental::stl

#include "suffix.hpp"
