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

#include "core/runtime/library.h"
#include "core/utilities/detail/type_traits.h"
#include "core/utilities/typedefs.h"

#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace legate {

class TaskInfo;

namespace detail {

template <typename T, template <typename...> typename SELECTOR, bool valid>
class VariantHelper;

class Runtime;

namespace cython {

void cytaskinfo_add_variant(legate::TaskInfo* handle,
                            legate::Library* core_lib,
                            legate::LegateVariantCode variant_kind,
                            legate::VariantImpl cy_entry,
                            legate::Processor::TaskFuncPtr py_entry);

}  // namespace cython

}  // namespace detail

class VariantInfo {
 public:
  VariantInfo() = default;

  static_assert(!traits::detail::is_pure_move_constructible_v<Legion::CodeDescriptor>,
                "Use by value and std::move for Legion::CodeDescriptor");
  VariantInfo(VariantImpl body_, const Legion::CodeDescriptor& code_desc_, VariantOptions options_)
    : body{body_}, code_desc{code_desc_}, options{options_}
  {
  }

  VariantImpl body{};
  Legion::CodeDescriptor code_desc{};
  VariantOptions options{};
};

class TaskInfo {
 public:
  explicit TaskInfo(std::string task_name);
  ~TaskInfo();

  [[nodiscard]] std::string_view name() const;

  [[nodiscard]] std::optional<std::reference_wrapper<const VariantInfo>> find_variant(
    LegateVariantCode vid) const;
  [[deprecated("since 24.09: use find_variant() directly")]] [[nodiscard]] bool has_variant(
    LegateVariantCode vid) const;

  void register_task(Legion::TaskID task_id);

  TaskInfo(const TaskInfo&)            = delete;
  TaskInfo& operator=(const TaskInfo&) = delete;
  TaskInfo(TaskInfo&&)                 = delete;
  TaskInfo& operator=(TaskInfo&&)      = delete;

  class AddVariantKey {
    AddVariantKey() = default;

    friend TaskInfo;
    friend void legate::detail::cython::cytaskinfo_add_variant(legate::TaskInfo*,
                                                               legate::Library*,
                                                               legate::LegateVariantCode,
                                                               legate::VariantImpl,
                                                               legate::Processor::TaskFuncPtr);
    template <typename T, template <typename...> typename SELECTOR, bool valid>
    friend class detail::VariantHelper;
  };

  // These are "private" insofar that the access key is private
  // NOLINTNEXTLINE(readability-identifier-naming)
  void add_variant_(AddVariantKey,
                    Library library,
                    LegateVariantCode vid,
                    VariantImpl body,
                    Processor::TaskFuncPtr entry,
                    const VariantOptions& default_options,
                    const std::map<LegateVariantCode, VariantOptions>& registration_options = {});

  // These are "private" insofar that the access key is private
  // NOLINTBEGIN(readability-identifier-naming)
  template <typename T>
  void add_variant_(AddVariantKey,
                    Library library,
                    LegateVariantCode vid,
                    LegionVariantImpl<T> body,
                    Processor::TaskFuncPtr entry,
                    const VariantOptions& default_options,
                    const std::map<LegateVariantCode, VariantOptions>& registration_options = {});
  // NOLINTEND(readability-identifier-naming)

  // TODO(wonchanl): remove once scalar extraction workaround is removed
  class RuntimeAddVariantKey {
    RuntimeAddVariantKey() = default;

    friend class detail::Runtime;
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  void add_variant_(RuntimeAddVariantKey,
                    Library core_lib,
                    LegateVariantCode vid,
                    const VariantOptions* callsite_options,
                    Legion::CodeDescriptor&& descr);

  [[deprecated("since 24.09: Use LegateTask and LegateTask::register_variants() instead")]] void
  add_variant(LegateVariantCode vid,
              VariantImpl body,
              const Legion::CodeDescriptor& code_desc,
              const VariantOptions& options);
  [[deprecated("since 24.09: Use LegateTask and LegateTask::register_variants() instead")]] void
  add_variant(LegateVariantCode vid,
              VariantImpl body,
              Processor::TaskFuncPtr entry,
              const std::map<LegateVariantCode, VariantOptions>& all_options = {});
  [[deprecated("since 24.09: Use LegateTask and LegateTask::register_variants() instead")]] void
  add_variant(LegateVariantCode vid,
              VariantImpl body,
              Processor::TaskFuncPtr entry,
              const VariantOptions& default_options,
              const std::map<LegateVariantCode, VariantOptions>& all_options = {});

 private:
  friend std::ostream& operator<<(std::ostream& os, const TaskInfo& info);

  class Impl;

  std::unique_ptr<Impl> impl_;
};

}  // namespace legate

#include "core/task/task_info.inl"
