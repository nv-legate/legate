/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/runtime/library.h>
#include <legate/task/task_info.h>
#include <legate/task/variant_helper.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/detail/zstring_view.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <map>

/**
 * @file
 * @brief Class definition for legate::LegateTask
 */

namespace legate {

class TaskConfig;

/**
 * @addtogroup task
 * @{
 */

/**
 * @brief A base class template for Legate task implementations.
 *
 * Any Legate task class must inherit legate::LegateTask directly or transitively. The type
 * parameter `T` needs to be bound to a child Legate task class that inherits legate::LegateTask.
 *
 * Currently, each task can have up to three variants. Each variant must be static member
 * functions of the class under the following names and signatures:
 *
 * - `void cpu_variant(legate::TaskContext)`: CPU implementation of the task
 * - `void gpu_variant(legate::TaskContext)`: GPU implementation of the task
 * - `void omp_variant(legate::TaskContext)`: OpenMP implementation of the task
 *
 * Tasks must have at least one variant, and all task variants must be semantically equivalent
 * (modulo some minor rounding errors due to floating point imprecision).
 *
 * Each task class must also have a type alias `Registrar` that points to a library specific
 * registrar class. (See legate::TaskRegistrar for details.)
 *
 * Tasks may also declare the following static members, which are used to populate defaults and
 * other information in various circumstances. These are split into 2 categories, task-wide and
 * per-variant. Task-wide members are usually used to describe invariants across the entire
 * task, or, to provide default values across all task variants. Per-variant members on the
 * other hand apply only to the specified variant, and override any task-wide settings where
 * applicable.
 *
 * Task-wide static members:
 *
 * - `static const legate::TaskConfig TASK_CONFIG`: This specifies the task-wide configuration,
 *   such as the task ID, the task signature, and default variant options to be used.
 *
 * Per-variant static members:
 *
 * - `static constexpr VariantOptions CPU_VARIANT_OPTIONS`: Specifies the default variant
 *   options used when registering the CPU variant of the task.
 * - `static constexpr VariantOptions OMP_VARIANT_OPTIONS`: Specifies the default variant
 *   options used when registering the OMP variant of the task.
 * - `static constexpr VariantOptions GPU_VARIANT_OPTIONS`: Specifies the default variant
 *   options used when registering the GPU variant of the task.
 *
 * If the default variant options are not present, the variant options for a given variant `v`
 * are selected in the following order:
 *
 * #. The variant options (if any) supplied at the call-site of `register_variants()`.
 * #. The default variant options (if any) found in `XXX_VARIANT_OPTIONS`.
 * #. (If defined) The default variant options found in `TASK_CONFIG` (if any).
 * #. The variant options provided by `Library::get_default_variant_options()`.
 * #. The global default variant options found in `VariantOptions::DEFAULT_OPTIONS`.
 *
 * @note Users are *highly* encouraged to use these static members to pre-declare their task
 * and variant properties. In all cases, the same information can be supplied dynamically at
 * either task registration, construction, or launch time, but doing so statically is preferred
 * as the runtime is able to make more efficient decisions when scheduling or launching the
 * tasks.
 *
 * @see TaskConfig
 * @see VariantOptions
 */
template <typename T>
class LegateTask {  // NOLINT(bugprone-crtp-constructor-accessibility)
 public:
  // Exports the base class so we can access it via subclass T
  using BASE = LegateTask<T>;

  /**
   * @brief Records all variants of this task in a registrar.
   *
   * Registers the variant with the task registrar (pointed to by the task's static type alias
   * `Registrar`, see `TaskRegistrar` for details about setting up a registrar in a library).
   *
   * The registration of the task is deferred until such time as
   * `TaskRegistrar::register_all_tasks()` is called.
   *
   * The task must have a static `TASK_CONFIG` member defined. Failure to do so is diagnosed at
   * compile-time.
   *
   * @param all_options Options for task variants. Variants with no entries in `all_options` will
   * use the default set of options as discussed in the class description.
   */
  static void register_variants(std::map<VariantCode, VariantOptions> all_options = {});

  /**
   * @brief Registers all variants of this task immediately.
   *
   * Registration of the task is performed immediately.
   *
   * The value of `T::TASK_CONFIG.task_id()` is used as the task id.
   *
   * @param library Library to which the task should be registered.
   * @param all_options Options for task variants. Variants with no entries in `all_options` will
   * use the default set of options as discussed in the class description.
   */
  static void register_variants(Library library,
                                const std::map<VariantCode, VariantOptions>& all_options = {});

  /**
   * @brief Registers all variants of this task immediately.
   *
   * Registration of the task is performed immediately.
   *
   * In almost all cases, the user should prefer the `TaskConfig` overload to this method, as
   * it allows specifying additional task properties.
   *
   * @param library Library to which the task should be registered.
   * @param task_id Task id.
   * @param all_options Options for task variants. Variants with no entries in `all_options` will
   * use the default set of options as discussed in the class description.
   */
  static void register_variants(Library library,
                                LocalTaskID task_id,
                                const std::map<VariantCode, VariantOptions>& all_options = {});

  /**
   * @brief Registers all variants of this task immediately.
   *
   * Registration of the task is performed immediately.
   *
   * @param library Library to which the task should be registered.
   * @param task_config The task configuration.
   * @param all_options Options for task variants. Variants with no entries in `all_options` will
   * use the default set of options as discussed in the class description.
   */
  static void register_variants(Library library,
                                const TaskConfig& task_config,
                                const std::map<VariantCode, VariantOptions>& all_options = {});

 protected:
  [[nodiscard]] static detail::ZStringView task_name_();

 private:
  template <typename, template <typename...> typename, bool>
  friend class detail::VariantHelper;

  // A helper to find and register all variants of a task
  [[nodiscard]] static TaskInfo create_task_info_(
    const Library& lib,
    const TaskConfig& task_config,
    const std::map<VariantCode, VariantOptions>& all_options);

  template <VariantImpl variant_fn, VariantCode variant_kind>
  static void task_wrapper_(const void* args,
                            std::size_t arglen,
                            const void* userdata,
                            std::size_t userlen,
                            Legion::Processor p);
};

/** @} */

}  // namespace legate

#include <legate/task/task.inl>
