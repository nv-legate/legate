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

// Useful for IDEs
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

template <typename T>
ConsensusMatchResult<T> Runtime::issue_consensus_match(std::vector<T>&& input)
{
  return {std::move(input), legion_context_, legion_runtime_};
}

template <typename T>
T Runtime::get_tunable(Legion::MapperID mapper_id, std::int64_t tunable_id)
{
  return get_tunable(mapper_id, tunable_id, sizeof(T)).get_result<T>();
}

template <typename T>
T Runtime::get_core_tunable(std::int64_t tunable_id)
{
  return get_tunable<T>(core_library_->get_mapper_id(), tunable_id);
}

inline bool Runtime::initialized() const { return initialized_; }

inline void Runtime::register_shutdown_callback(ShutdownCallback callback)
{
  callbacks_.emplace_back(std::move(callback));
}

inline const Library* Runtime::core_library() const { return core_library_; }

inline std::uint64_t Runtime::current_op_id() const { return current_op_id_; }

inline void Runtime::increment_op_id() { ++current_op_id_; }

inline std::uint64_t Runtime::get_unique_store_id() { return next_store_id_++; }

inline std::uint64_t Runtime::get_unique_storage_id() { return next_storage_id_++; }

inline std::uint32_t Runtime::field_reuse_freq() const { return field_reuse_freq_; }

inline PartitionManager* Runtime::partition_manager() const { return partition_manager_.get(); }

inline CommunicatorManager* Runtime::communicator_manager() const
{
  return communicator_manager_.get();
}

inline Scope& Runtime::scope() { return scope_; }

inline const Scope& Runtime::scope() const { return scope_; }

inline const mapping::detail::LocalMachine& Runtime::local_machine() const
{
  return local_machine_;
}

}  // namespace legate::detail
