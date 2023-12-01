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

#include "core/operation/detail/task.h"

namespace legate::detail {

inline Task::ArrayArg::ArrayArg(std::shared_ptr<LogicalArray> _array) : array{std::move(_array)} {}

inline Task::ArrayArg::ArrayArg(std::shared_ptr<LogicalArray> _array,
                                std::optional<SymbolicPoint> _projection)
  : array{std::move(_array)}, projection{std::move(_projection)}
{
}

// ==========================================================================================

inline bool Task::always_flush() const { return can_throw_exception_; }

// ==========================================================================================

inline AutoTask::AutoTask(const Library* library,
                          int64_t task_id,
                          uint64_t unique_id,
                          mapping::detail::Machine&& machine)
  : Task{library, task_id, unique_id, std::move(machine)}
{
}

// ==========================================================================================

inline void ManualTask::validate() {}

inline void ManualTask::launch(Strategy* /*strategy*/) { launch(); }

inline void ManualTask::launch() { launch_task(strategy_.get()); }

inline void ManualTask::add_to_solver(ConstraintSolver& /*solver*/) {}

}  // namespace legate::detail
