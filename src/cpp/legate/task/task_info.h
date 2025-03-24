/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/variant_info.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <string_view>

/** @file */

namespace legate {

class Library;
class TaskInfo;
class TaskConfig;

}  // namespace legate

namespace legate::detail {

class TaskInfo;

template <typename T, template <typename...> typename SELECTOR, bool valid>
class VariantHelper;

}  // namespace legate::detail

namespace legate::detail::cython {

void cytaskinfo_add_variant(legate::TaskInfo* handle,
                            const legate::Library&,
                            legate::VariantCode variant_kind,
                            legate::VariantImpl cy_entry,
                            legate::Processor::TaskFuncPtr py_entry,
                            const legate::TaskConfig& config);

}  // namespace legate::detail::cython

namespace legate {

/**
 * @addtogroup task
 * @{
 */

/**
 * @brief An object describing a Legate task registration info.
 */
class TaskInfo {
 public:
  TaskInfo() = LEGATE_DEFAULT_WHEN_CYTHON;

  /**
   * @brief Construct a `TaskInfo`.
   *
   * @param task_name The name of the task.
   */
  explicit TaskInfo(std::string task_name);

  /**
   * @brief Construct a `TaskInfo`.
   *
   * @param impl A pointer to the implementation class.
   */
  explicit TaskInfo(InternalSharedPtr<detail::TaskInfo> impl);

  /**
   * @return The name of the task.
   */
  [[nodiscard]] std::string_view name() const;

  /**
   * @brief Look up a variant of the task.
   *
   * @param vid The variant to look up.
   *
   * @return An optional containing the `VariantInfo` for the variant, or `std::nullopt` if the
   * variant was not found.
   *
   * @see VariantInfo
   */
  [[nodiscard]] std::optional<VariantInfo> find_variant(VariantCode vid) const;

  class AddVariantKey {
    AddVariantKey() = default;

    friend TaskInfo;
    friend void legate::detail::cython::cytaskinfo_add_variant(legate::TaskInfo*,
                                                               const legate::Library&,
                                                               legate::VariantCode,
                                                               legate::VariantImpl,
                                                               legate::Processor::TaskFuncPtr,
                                                               const legate::TaskConfig&);
    template <typename T, template <typename...> typename SELECTOR, bool valid>
    friend class detail::VariantHelper;
  };

  // These are "private" insofar that the access key is private

  /**
   * @brief Register a new variant to the task description.
   *
   * @param library The library to retrieve the default variant options from.
   * @param vid The variant type to register.
   * @param body The variant function pointer.
   * @param entry The pointer to the entry point wrapping `body`, to be passed to Legion.
   * @param task_config The task-wide configuration options.
   * @param decl_options Any variant options declared in the task declaration, or `nullptr` if
   * none were found.
   * @param registration_options Variant options specified at task registration time.
   */
  void add_variant_(AddVariantKey,  // NOLINT(readability-identifier-naming)
                    const Library& library,
                    VariantCode vid,
                    VariantImpl body,
                    Processor::TaskFuncPtr entry,
                    const TaskConfig& task_config,
                    const VariantOptions* decl_options,
                    const std::map<VariantCode, VariantOptions>& registration_options = {});

  // These are "private" insofar that the access key is private
  /**
   * @brief Register a new variant to the task description.
   *
   * @param library The library to retrieve the default variant options from.
   * @param vid The variant type to register.
   * @param body The variant function pointer.
   * @param entry The pointer to the entry point wrapping `body`, to be passed to Legion.
   * @param task_config The task-wide configuration options.
   * @param decl_options Any variant options declared in the task declaration, or `nullptr` if
   * none were found.
   * @param registration_options Variant options specified at task registration time.
   */
  template <typename T>
  void add_variant_(AddVariantKey,  // NOLINT(readability-identifier-naming)
                    const Library& library,
                    VariantCode vid,
                    LegionVariantImpl<T> body,
                    Processor::TaskFuncPtr entry,
                    const TaskConfig& task_config,
                    const VariantOptions* decl_options,
                    const std::map<VariantCode, VariantOptions>& registration_options = {});

  /**
   * @return A human-readable representation of the Task.
   */
  [[nodiscard]] std::string to_string() const;

  /**
   * @return The private implementation pointer.
   */
  [[nodiscard]] const SharedPtr<detail::TaskInfo>& impl() const;

 private:
  friend std::ostream& operator<<(std::ostream& os, const TaskInfo& info);

  SharedPtr<detail::TaskInfo> impl_{};
};

/** @} */

}  // namespace legate

#include <legate/task/task_info.inl>
