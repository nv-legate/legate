/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/scope.h>

#include <legate/runtime/detail/runtime.h>

#include <utility>

namespace legate::detail {

ParallelPolicy Scope::exchange_parallel_policy(ParallelPolicy parallel_policy)
{
  if (parallel_policy != parallel_policy_) {
    auto&& rt = Runtime::get_runtime();

    if (parallel_policy.streaming() || parallel_policy_.streaming()) {
      // Note, we want to issue this mapping fence if either the incoming or outgoing scope are
      // streaming, because during scheduling window flushes, the streaming generation will
      // change. See discussion in `BaseMapping::select_streaming_tasks_to_map()` for why this
      // fence is needed.
      rt.issue_mapping_fence();
    }
    rt.flush_scheduling_window();
  }
  return std::exchange(parallel_policy_, std::move(parallel_policy));
}

}  // namespace legate::detail
