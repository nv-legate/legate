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
    Runtime::get_runtime().flush_scheduling_window();
  }
  return std::exchange(parallel_policy_, std::move(parallel_policy));
}

}  // namespace legate::detail
