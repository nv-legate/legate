/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/detail/doxygen.h>

#include <array>
#include <optional>

/**
 * @file
 * @brief Definition for legate::ProcLocalStorage
 */

namespace legate {

/**
 * @addtogroup util
 * @{
 */

/**
 * @brief A helper data structure to store processor-local objects.
 *
 * Oftentimes, users need to create objects, usually some library handles, each of which is
 * associated with only one processor (GPU, most likely). For those cases, users can create a
 * `ProcLocalStorage<T>` that holds a unique singleton object of type `T` for each processor
 * thread. The object can be retrieved simply by the `get()` method and internally the calls are
 * distinguished by IDs of the processors invoking them.
 *
 * Two parallel tasks running on the same processor will get the same object if they query the same
 * `ProcLocalStorage`. Atomicity of access to the storage is guaranteed by the programming model
 * running parallel tasks atomically on each processor; in other words, no synchronization is needed
 * to call the `get()` method on a `ProcLocalStorage` even when it's shared by multiple tasks.
 *
 * Despite the name, the values that are stored in this storage don't have static storage duration,
 * but they are alive only as long as the owning `ProcLocalStorage` object is.
 *
 * This example uses a `ProcLocalStorage<int>` to count the number of task invocations on each
 * processor:
 *
 * @code{.cpp}
 * static void cpu_variant(legate::TaskContext context)
 * {
 *   static legate::ProcLocalStorage<int> counter{};
 *   if (!storage.has_value()) {
 *     // If this is the first visit, initialize the counter
 *     counter.emplace(1);
 *   } else {
 *     // Otherwise, increment the counter by 1
 *     ++counter.get();
 *   }
 * }
 * @endcode
 *
 * @tparam T Type of values stored in this `ProcLocalStorage`.
 */
template <typename T>
class ProcLocalStorage {
 public:
  /**
   * @brief The type of stored objects.
   */
  using value_type = T;

  /**
   * @brief Checks if the value has been created for the executing processor.
   *
   * @return `true` if the value exists, `false` otherwise.
   */
  [[nodiscard]] bool has_value() const;

  /**
   * @brief Constructs a new value for the executing processor.
   *
   * The existing value will be overwritten by the new value.
   *
   * @param args Arguments to the constructor of type `T`.
   */
  template <typename... Args>
  value_type& emplace(Args&&... args);

  /**
   * @brief Returns the value for the executing processor.
   *
   * @return the value for the executing processor.
   *
   * @throws std::logic_error If no value exists for this processor (i.e., if `has_value()` returns
   * `false`), or if the method is invoked outside a task
   */
  [[nodiscard]] constexpr value_type& get();

  /**
   * @brief Returns the value for the executing processor.
   *
   * @return the value for the executing processor
   *
   * @throws std::logic_error If no value exists for this processor (i.e., if `has_value()` returns
   * `false`), or if the method is invoked outside a task
   */
  [[nodiscard]] constexpr const value_type& get() const;

 private:
  std::array<std::optional<T>, LEGATE_MAX_NUM_PROCS> storage_{};
};

/** @} */

}  // namespace legate

#include <legate/utilities/proc_local_storage.inl>
