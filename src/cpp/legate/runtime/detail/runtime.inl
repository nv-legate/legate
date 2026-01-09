/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Useful for IDEs
#include <legate/runtime/detail/runtime.h>

namespace legate::detail {

inline const Config& Runtime::config() const { return config_; }

template <typename T>
ConsensusMatchResult<T> Runtime::issue_consensus_match(std::vector<T>&& input)
{
  return {std::move(input), get_legion_context(), get_legion_runtime()};
}

inline bool Runtime::initialized() const { return initialized_; }

inline const Library& Runtime::core_library() const
{
  return core_library_.value();  // NOLINT(bugprone-unchecked-optional-access)
}

inline Legion::Runtime* Runtime::get_legion_runtime() { return legion_runtime_; }

inline Legion::Context Runtime::get_legion_context() { return legion_context_; }

inline std::uint64_t Runtime::new_op_id() { return ++cur_op_id_; }

inline std::uint64_t Runtime::get_unique_store_id() { return next_store_id_++; }

inline std::uint64_t Runtime::get_unique_storage_id() { return next_storage_id_++; }

inline std::uint32_t Runtime::field_reuse_freq() const { return config().field_reuse_freq(); }

inline std::size_t Runtime::field_reuse_size() const { return field_reuse_size_; }

inline FieldManager& Runtime::field_manager() { return *field_manager_; }

inline PartitionManager& Runtime::partition_manager()
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    return partition_manager_.value();  // NOLINT(bugprone-unchecked-optional-access)
  }
  return *partition_manager_;  // NOLINT(bugprone-unchecked-optional-access)
}

inline const PartitionManager& Runtime::partition_manager() const
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    return partition_manager_.value();  // NOLINT(bugprone-unchecked-optional-access)
  }
  return *partition_manager_;  // NOLINT(bugprone-unchecked-optional-access)
}

inline CommunicatorManager& Runtime::communicator_manager()
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    return communicator_manager_.value();  // NOLINT(bugprone-unchecked-optional-access)
  }
  return *communicator_manager_;  // NOLINT(bugprone-unchecked-optional-access)
}

inline const CommunicatorManager& Runtime::communicator_manager() const
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    return communicator_manager_.value();  // NOLINT(bugprone-unchecked-optional-access)
  }
  return *communicator_manager_;  // NOLINT(bugprone-unchecked-optional-access)
}

inline Scope& Runtime::scope() { return scope_; }

inline const Scope& Runtime::scope() const { return scope_; }

inline const mapping::detail::LocalMachine& Runtime::local_machine() const
{
  return local_machine_;
}

inline std::uint32_t Runtime::node_count() const { return local_machine().total_nodes; }

inline std::uint32_t Runtime::node_id() const { return local_machine().node_id; }

inline bool Runtime::executing_inline_task() const noexcept { return executing_inline_task_; }

inline void Runtime::inline_task_start() noexcept { executing_inline_task_ = true; }

inline void Runtime::inline_task_end() noexcept
{
  LEGATE_ASSERT(executing_inline_task_);
  executing_inline_task_ = false;
}

}  // namespace legate::detail
