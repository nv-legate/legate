/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>

#include <cstdint>
#include <functional>
#include <memory>

/**
 * @file
 * @brief Class definition for legate::TaskRegistrar
 */

namespace legate {

enum class LocalTaskID : std::int64_t;

template <typename T>
class LegateTask;  // NOLINT(bugprone-crtp-constructor-accessibility)
class TaskInfo;
class Library;

/**
 * @addtogroup task
 * @{
 */

/**
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
  TaskRegistrar();
  ~TaskRegistrar();

  TaskRegistrar(TaskRegistrar&&)            = delete;
  TaskRegistrar& operator=(TaskRegistrar&&) = delete;

  /**
   * @brief Registers all tasks recorded in this registrar.
   *
   * This function is typically called in the task registration callback of a library
   * and must be called after the library is fully initialized.
   *
   * @param library Library that owns this registrar.
   */
  void register_all_tasks(Library& library);

  class RecordTaskKey {
    RecordTaskKey() = default;

    friend TaskRegistrar;
    template <typename T>
    friend class LegateTask;
  };

  /*
   * Record a function to be called later (only to be used via `Task::register_variants()`).
   */
  void record_registration_function(RecordTaskKey key, std::function<void(const Library&)> func);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

/** @} */

}  // namespace legate
