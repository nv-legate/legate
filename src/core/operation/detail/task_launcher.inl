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

#include "core/operation/detail/task_launcher.h"

namespace legate::detail {

inline TaskLauncher::TaskLauncher(const Library* library,
                                  const mapping::detail::Machine& machine,
                                  std::variant<std::string_view, std::string> provenance,
                                  LocalTaskID task_id,
                                  Legion::MappingTagID tag)
  : library_{library},
    task_id_{task_id},
    tag_{tag},
    machine_{machine},
    provenance_{[](std::variant<std::string_view, std::string> p)
                  -> std::variant<std::string_view, std::string> {
      if (const auto* sv_ptr = std::get_if<std::string_view>(&p)) {
        // If the string_view is empty, we convert to std::string. We do this because
        // std::string.c_str() for an empty string still returns a valid, (null terminated!) c
        // string (""). Notably, this is still maximally performant since the empty string will
        // not allocate.
        //
        // If the string_view isn't null-terminated then we also want to convert to std::string
        // since -- once again -- std::string.c_str() will then do the right thing for us.
        if (sv_ptr->empty() || sv_ptr->back()) {
          return std::string{*sv_ptr};
        }
      }
      return p;
    }(std::move(provenance))}
{
}

inline TaskLauncher::TaskLauncher(const Library* library,
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

inline std::string_view TaskLauncher::provenance() const
{
  return std::visit([](const auto& p) -> std::string_view { return p; }, provenance_);
}

}  // namespace legate::detail
