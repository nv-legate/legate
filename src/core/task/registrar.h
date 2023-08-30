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

#include <memory>

#include "legion.h"

#include "core/runtime/library.h"
#include "core/task/variant_options.h"
#include "core/utilities/typedefs.h"

/**
 * @file
 * @brief Class definition fo legate::TaskRegistrar
 */

namespace legate {

class TaskInfo;

/**
 * @ingroup task
 * @brief A helper class for task variant registration.
 *
 * The `legate::TaskRegistrar` class is designed to simplify the boilerplate that client libraries
 * need to register all its task variants. The following is a boilerplate that each library
 * needs to write:
 *
 * @code{.cpp}
 * struct MyLibrary {
 *   static legate::TaskRegistrar& get_registrar();
 * };
 *
 * template <typename T>
 * struct MyLibraryTaskBase : public legate::LegateTask<T> {
 *   using Registrar = MyLibrary;
 *
 *   ...
 * };
 * @endcode
 *
 * In the code above, the `MyLibrary` has a static member that returns a singleton
 * `legate::TaskRegistrar` object. Then, the `MyLibraryTaskBase` points to the class so Legate can
 * find where task variants are collected.
 *
 * Once this registrar is set up in a library, each library task can simply register itself
 * with the `LegateTask::register_variants` method like the following:
 *
 * @code{.cpp}
 * // In a header
 * struct MyLibraryTask : public MyLibraryTaskBase<MyLibraryTask> {
 *   ...
 * };
 *
 * // In a C++ file
 * static void __attribute__((constructor)) register_tasks()
 * {
 *   MyLibraryTask::register_variants();
 * }
 * @endcode
 */
class TaskRegistrar {
 public:
  /**
   * @brief Registers all tasks recorded in this registrar. Typically invoked in a registration
   * callback of a library.
   *
   * @param library Library that owns this registrar
   */
  void register_all_tasks(Library library);

 public:
  void record_task(int64_t local_task_id, std::unique_ptr<TaskInfo> task_info);

 public:
  TaskRegistrar();
  ~TaskRegistrar();

 private:
  TaskRegistrar(const TaskRegistrar&)            = delete;
  TaskRegistrar& operator=(const TaskRegistrar&) = delete;
  TaskRegistrar(TaskRegistrar&&)                 = delete;
  TaskRegistrar& operator=(TaskRegistrar&&)      = delete;

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace legate
