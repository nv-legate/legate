/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/internal_shared_ptr.h>

#include <legion.h>

#include <optional>
#include <unordered_map>
#include <variant>

namespace legate::detail {

class LogicalStore;
class PhysicalStore;
class Variable;

class TaskStoreArg {
 public:
  /**
   * @brief Construct a TaskStoreArg for LogicalStore.
   *
   * `priv` should be initialized to the following values based on whether the argument as an
   * input, output, or reduction:
   *
   * - input: `LEGION_READ_ONLY`
   * - output: `LEGION_WRITE_ONLY`
   * - reduction: `LEGION_REDUCE`
   *
   * If the owning task is a streaming task, then this privilege is further fixed up during
   * scheduling window flush to include additional discard privileges. Therefore, the privilege
   * member should *not* be considered stable until the task is sent to Legion.
   *
   * @param priv The access privilege for this task argument.
   * @param _store The LogicalStore for this argument.
   */
  TaskStoreArg(Legion::PrivilegeMode priv,
               InternalSharedPtr<LogicalStore> _store,
               const Variable* _variable = nullptr);

  /**
   * @brief Construct a TaskStoreArg for PhysicalStore.
   *
   * @param priv The access privilege for this task argument.
   * @param _store The PhysicalStore for this argument.
   */
  TaskStoreArg(Legion::PrivilegeMode priv, InternalSharedPtr<PhysicalStore> _store);
  [[nodiscard]] bool needs_flush() const;

  Legion::PrivilegeMode privilege{Legion::PrivilegeMode::LEGION_NO_ACCESS};
  std::variant<InternalSharedPtr<LogicalStore>, InternalSharedPtr<PhysicalStore>> store{};
  const Variable* variable{};
};

}  // namespace legate::detail
