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

#include "core/operation/detail/task_launcher.h"

namespace legate::detail {

inline TaskLauncher::TaskLauncher(const Library* library,
                                  const mapping::detail::Machine& machine,
                                  std::string provenance,
                                  std::int64_t task_id,
                                  std::int64_t tag)
  : library_{library},
    task_id_{task_id},
    tag_{tag},
    machine_{machine},
    provenance_{std::move(provenance)}
{
}

inline TaskLauncher::TaskLauncher(const Library* library,
                                  const mapping::detail::Machine& machine,
                                  std::int64_t task_id,
                                  std::int64_t tag)
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

inline void TaskLauncher::throws_exception(bool can_throw_exception)
{
  can_throw_exception_ = can_throw_exception;
}

inline void TaskLauncher::relax_interference_checks(bool relax)
{
  relax_interference_checks_ = relax;
}

}  // namespace legate::detail
