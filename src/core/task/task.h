/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "core/runtime/library.h"
#include "core/task/task_info.h"
#include "core/task/variant_helper.h"
#include "core/task/variant_options.h"
#include "core/utilities/typedefs.h"

#include <map>
#include <memory>
#include <string>

/** @defgroup task Task
 */

/**
 * @file
 * @brief Class definition fo legate::LegateTask
 */
namespace legate {

/**
 * @ingroup task
 * @brief A base class template for Legate task implementations.
 *
 * Any Legate task class must inherit legate::LegateTask directly or transitively. The type
 * parameter `T` needs to be bound to a child Legate task class that inherits legate::LegateTask.
 *
 * Curently, each task can have up to three variants and the variants need to be static member
 * functions of the class under the following names:
 *
 *   - `cpu_variant`: CPU implementation of the task
 *   - `gpu_variant`: GPU implementation of the task
 *   - `omp_variant`: OpenMP implementation of the task
 *
 * Tasks must have at least one variant, and all task variants must be semantically equivalent
 * (modulo some minor rounding errors due to floating point imprecision).
 *
 * Each task class must also have a type alias `Registrar` that points to a library specific
 * registrar class. (See legate::TaskRegistrar for details.)
 */
template <typename T>
struct LegateTask {
  // Exports the base class so we can access it via subclass T
  using BASE = LegateTask<T>;

  /**
   * @brief Records all variants of this task in a registrar.
   *
   * The registrar is pointed to by the task's static type alias `Registrar` (see
   * legate::TaskRegistrar for details about setting up a registrar in a library). The client
   * can optionally specify variant options.
   *
   * @param all_options Options for task variants. Variants with no entires in `all_options` will
   * use the default set of options
   */
  static void register_variants(
    const std::map<LegateVariantCode, VariantOptions>& all_options = {});

  /**
   * @brief Registers all variants of this task immediately.
   *
   * Unlike the other method, this one takes a library so the registration can be done immediately.
   * The value of T::TASK_ID is used as the task id.
   *
   * @param library Library to which the task should be registered
   * @param all_options Options for task variants. Variants with no entires in `all_options` will
   * use the default set of options
   */
  static void register_variants(
    Library library, const std::map<LegateVariantCode, VariantOptions>& all_options = {});

  /**
   * @brief Registers all variants of this task immediately.
   *
   * Unlike the other method, this one takes a library so the registration can be done immediately.
   *
   * @param library Library to which the task should be registered
   * @param task_id Task id
   * @param all_options Options for task variants. Variants with no entires in `all_options` will
   * use the default set of options
   */
  static void register_variants(
    Library library,
    int64_t task_id,
    const std::map<LegateVariantCode, VariantOptions>& all_options = {});

 private:
  template <typename, template <typename...> typename, bool>
  friend struct detail::VariantHelper;

  // A wrapper that wraps all Legate task variant implementations. Provides
  // common functionalities and instrumentations
  template <VariantImpl VARIANT_IMPL>
  static void legate_task_wrapper(
    const void* args, size_t arglen, const void* userdata, size_t userlen, Processor p);

  // A helper to find and register all variants of a task
  static std::unique_ptr<TaskInfo> create_task_info(
    const std::map<LegateVariantCode, VariantOptions>& all_options);

  [[nodiscard]] static const std::string& task_name();
};

}  // namespace legate

#include "core/task/task.inl"
