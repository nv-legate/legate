/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/projection.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <optional>
#include <unordered_map>
#include <variant>

namespace legate::detail {

class LogicalArray;
class LogicalStore;
class PhysicalArray;
class Variable;

class TaskArrayArg {
 public:
  /**
   * @brief Construct a TaskArrayArg for LogicalArray.
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
   * @param _array The LogicalArray for this argument.
   * @param _projection An optional projection for the argument.
   */
  TaskArrayArg(Legion::PrivilegeMode priv,
               InternalSharedPtr<LogicalArray> _array,
               std::optional<SymbolicPoint> _projection = std::nullopt);

  /**
   * @brief Construct a TaskArrayArg for PhysicalArray.
   *
   * @param priv The access privilege for this task argument.
   * @param _array The PhysicalArray for this argument.
   */
  TaskArrayArg(Legion::PrivilegeMode priv, InternalSharedPtr<PhysicalArray> _array);
  [[nodiscard]] bool needs_flush() const;

  Legion::PrivilegeMode privilege{Legion::PrivilegeMode::LEGION_NO_ACCESS};
  std::variant<InternalSharedPtr<LogicalArray>, InternalSharedPtr<PhysicalArray>> array{};
  std::unordered_map<InternalSharedPtr<LogicalStore>, const Variable*>
    mapping{};  // Only used for LogicalArray
  std::optional<SymbolicPoint> projection{};
};

}  // namespace legate::detail
