/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/detail/doxygen.h>

#include <legion.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

/**
 * @file
 * @brief Type aliases to Legion components
 */

namespace legate {

class TaskContext;

/**
 * @addtogroup util
 * @{
 */

/**
 * @brief Function signature for task variants. Each task variant must be a function of this type.
 */
using VariantImpl = void (*)(TaskContext);

/**
 * @brief Signature for a callable to be executed right before the runtime shuts down.
 */
using ShutdownCallback = std::function<void(void)>;

/**
 * @brief An enum describing the kind of variant.
 *
 * @note The values don't start at 0. This is to match Legion, where `0` is the 'None' variant.
 */
enum class VariantCode : std::uint32_t {  // NOLINT(performance-enum-size)
  CPU = 1,                                ///< A CPU variant.
  GPU,                                    ///< A GPU variant.
  OMP                                     ///< An OpenMP variant.
};

using Realm::Logger;

/**
 * @brief Integer type representing a `Library`-local task ID.
 *
 * All tasks in a `Library` are uniquely identifiable via a "task ID". When a task is registered to
 * a `Library`, the task must declare a unique "local" task ID (`LocalTaskID`) within that
 * `Library`.  This task ID must not coincide with any other task ID within that `Library`.
 *
 * For example, consider a task `Foo`:
 * @snippet unit/library.cc Foo declaration
 * And a `Library` `bar`:
 * @snippet unit/library.cc TaskID registration
 *
 * @see Library
 */
enum class LocalTaskID : std::int64_t {};

/**
 * @brief Integer type representing a `Library`-local reduction operator ID.
 *
 * All reduction operators are uniquely identifiable via a "reduction ID", which serve as proxy
 * task ID's for the reduction meta-tasks. When a reduction operator is registered with a
 * `Library`, the reduction must declare a unique "local" ID (`LocalRedopID`) within that
 * `Library`. The `Library` then assigns a globally unique ID to the reduction operator, which
 * may be used to refer to the operator across the entire program.
 *
 * @see GlobalRedopID Library Library::get_reduction_op_id()
 */
enum class LocalRedopID : std::int64_t {};

/**
 * @brief Integer type representing a global reduction operator ID.
 *
 * `GlobalRedopID`s may be used to refer to reduction operators registered to other `Library`s,
 * or to refer to the reduction operator when interfacing with Legion. See `LocalRedopID` for
 * further discussion on reduction operator ID's.
 *
 * @see LocalRedopID
 */
enum class GlobalRedopID : Legion::ReductionOpID {};

/** @} */  // end of util

// Geometry types

/**
 * @addtogroup geometry
 * @{
 */

/**
 * @brief Coordinate type.
 */
using coord_t = std::int64_t;

// `legate::NDPoint` and `legate::NDRect` are defined natively in <legate/geometry/nd_point.h> and
// <legate/geometry/nd_rect.h> (included above).

/** @} */  // end of geometry

// Iterators

/**
 * @addtogroup iterator
 * @{
 */

// To iterate over the points of a `legate::NDRect`, use `legate::NDRect::iterate()`
// (see <legate/geometry/nd_rect.h>).

/** @} */  // end of iterator

using Realm::Memory;
using Realm::Processor;

namespace detail {

[[nodiscard]] LEGATE_EXPORT Logger& log_legate();

[[nodiscard]] LEGATE_EXPORT Logger& log_legate_partitioner();

}  // namespace detail

namespace cuda::detail {

[[nodiscard]] LEGATE_EXPORT Logger& logger();

}  // namespace cuda::detail

}  // namespace legate
