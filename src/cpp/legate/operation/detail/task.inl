/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/task.h>
#include <legate/runtime/detail/config.h>

namespace legate::detail {

inline TaskArrayArg::TaskArrayArg(InternalSharedPtr<LogicalArray> _array) : array{std::move(_array)}
{
}

inline TaskArrayArg::TaskArrayArg(InternalSharedPtr<LogicalArray> _array,
                                  std::optional<SymbolicPoint> _projection)
  : array{std::move(_array)}, projection{std::move(_projection)}
{
}

inline bool TaskArrayArg::needs_flush() const { return array->needs_flush(); }

// ==========================================================================================

inline bool Task::supports_replicated_write() const { return true; }

inline bool Task::can_throw_exception() const { return can_throw_exception_; }

inline bool Task::can_elide_device_ctx_sync() const { return can_elide_device_ctx_sync_; }

inline const Library& Task::library() const { return library_; }

inline LocalTaskID Task::local_task_id() const { return task_id_; }

inline const std::vector<InternalSharedPtr<Scalar>>& Task::scalars() const { return scalars_; }

inline const std::vector<TaskArrayArg>& Task::inputs() const { return inputs_; }

inline const std::vector<TaskArrayArg>& Task::outputs() const { return outputs_; }

inline const std::vector<TaskArrayArg>& Task::reductions() const { return reductions_; }

inline const std::vector<InternalSharedPtr<LogicalStore>>& Task::scalar_outputs() const
{
  return scalar_outputs_;
}

inline const std::vector<std::pair<InternalSharedPtr<LogicalStore>, GlobalRedopID>>&
Task::scalar_reductions() const
{
  return scalar_reductions_;
}

inline const VariantInfo& Task::variant_info_() const { return vinfo_; }

// ==========================================================================================

inline Operation::Kind AutoTask::kind() const { return Kind::AUTO_TASK; }

inline bool AutoTask::needs_partitioning() const { return true; }

// ==========================================================================================

inline Operation::Kind ManualTask::kind() const { return Kind::MANUAL_TASK; }

inline bool ManualTask::needs_partitioning() const { return false; }

}  // namespace legate::detail
