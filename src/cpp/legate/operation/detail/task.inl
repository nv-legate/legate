/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/task.h>
#include <legate/utilities/detail/type_traits.h>

namespace legate::detail {

// ==========================================================================================

inline bool TaskBase::supports_replicated_write() const { return true; }

inline bool TaskBase::can_throw_exception() const { return can_throw_exception_; }

inline bool TaskBase::can_elide_device_ctx_sync() const { return can_elide_device_ctx_sync_; }

inline const Library& TaskBase::library() const { return library_; }

inline LocalTaskID TaskBase::local_task_id() const { return task_id_; }

inline Span<const InternalSharedPtr<Scalar>> TaskBase::scalars() const { return scalars_; }

inline Span<const TaskArrayArg> TaskBase::inputs() const { return inputs_; }

inline Span<const TaskArrayArg> TaskBase::outputs() const { return outputs_; }

inline Span<const TaskArrayArg> TaskBase::reductions() const { return reductions_; }

inline const VariantInfo& TaskBase::variant_info_() const { return vinfo_; }

// ==========================================================================================

inline const std::optional<StreamingGeneration>& LogicalTask::streaming_generation() const
{
  return streaming_gen_;
}

inline Span<const InternalSharedPtr<LogicalStore>> LogicalTask::scalar_outputs() const
{
  return scalar_outputs_;
}

inline Span<const std::pair<InternalSharedPtr<LogicalStore>, GlobalRedopID>>
LogicalTask::scalar_reductions() const
{
  return scalar_reductions_;
}

// ==========================================================================================

inline Operation::Kind AutoTask::kind() const { return Kind::AUTO_TASK; }

inline bool AutoTask::needs_partitioning() const { return true; }

// ==========================================================================================

inline Operation::Kind ManualTask::kind() const { return Kind::MANUAL_TASK; }

inline bool ManualTask::needs_partitioning() const { return false; }

inline bool ManualTask::supports_streaming() const { return true; }

inline const InternalSharedPtr<Strategy>& ManualTask::strategy() const { return strategy_; }

// ==========================================================================================

inline Operation::Kind PhysicalTask::kind() const { return Kind::PHYSICAL_TASK; }

inline bool PhysicalTask::needs_partitioning() const { return false; }

inline Span<const InternalSharedPtr<PhysicalStore>> PhysicalTask::physical_scalar_outputs() const
{
  return scalar_outputs_;
}

inline Span<const std::pair<InternalSharedPtr<PhysicalStore>, GlobalRedopID>>
PhysicalTask::physical_scalar_reductions() const
{
  return scalar_reductions_;
}

}  // namespace legate::detail
