/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/task_launcher.h>

namespace legate::detail {

inline TaskLauncher::TaskLauncher(const Library& library,
                                  const mapping::detail::Machine& machine,
                                  ZStringView provenance,
                                  LocalTaskID task_id,
                                  Legion::MappingTagID tag)
  : library_{library},
    task_id_{task_id},
    tag_{tag},
    machine_{machine},
    provenance_{std::move(provenance)}
{
}

inline TaskLauncher::TaskLauncher(const Library& library,
                                  const mapping::detail::Machine& machine,
                                  LocalTaskID task_id,
                                  Legion::MappingTagID tag)
  : TaskLauncher{library, machine, {}, task_id, tag}
{
}

inline void TaskLauncher::set_priority(std::int32_t priority) { priority_ = priority; }

inline void TaskLauncher::set_side_effect(bool has_side_effect)
{
  has_side_effect_ = has_side_effect;
}

inline void TaskLauncher::set_concurrent(bool is_concurrent) { concurrent_ = is_concurrent; }

inline void TaskLauncher::set_insert_barrier(bool insert_barrier)
{
  insert_barrier_ = insert_barrier;
}

inline void TaskLauncher::set_future_size(std::size_t future_size) { future_size_ = future_size; }

inline void TaskLauncher::throws_exception(bool can_throw_exception)
{
  can_throw_exception_ = can_throw_exception;
}

inline void TaskLauncher::can_elide_device_ctx_sync(bool can_elide_sync)
{
  can_elide_device_ctx_sync_ = can_elide_sync;
}

inline void TaskLauncher::relax_interference_checks(bool relax)
{
  relax_interference_checks_ = relax;
}

inline ZStringView TaskLauncher::provenance() const { return provenance_; }

}  // namespace legate::detail
