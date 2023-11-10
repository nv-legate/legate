/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

// Useful for IDEs
#include "core/runtime/detail/runtime.h"

namespace legate::detail {

template <typename T>
ConsensusMatchResult<T>::ConsensusMatchResult(std::vector<T>&& input,
                                              Legion::Context ctx,
                                              Legion::Runtime* runtime)
  : input_{std::move(input)},
    output_{input_.size()},
    future_{runtime->consensus_match(
      std::move(ctx), input_.data(), output_.data(), input_.size(), sizeof(T))}
{
}

template <typename T>
ConsensusMatchResult<T>::~ConsensusMatchResult() noexcept
{
  // Make sure the consensus match operation has completed, because it will be scribbling over the
  // buffers in this object.
  if (!future_.valid()) return;
  try {
    wait();
  } catch (const std::exception& excn) {
    log_legate().error() << excn.what();
    LEGATE_ABORT;
  }
}

template <typename T>
void ConsensusMatchResult<T>::wait()
{
  if (complete_) return;
  const auto num_matched = future_.get_result<std::size_t>();
  assert(num_matched <= output_.size());
  output_.resize(num_matched);
  complete_ = true;
};

template <typename T>
const std::vector<T>& ConsensusMatchResult<T>::input() const
{
  return input_;
};

template <typename T>
const std::vector<T>& ConsensusMatchResult<T>::output() const
{
  assert(complete_);
  return output_;
};

template <typename T>
ConsensusMatchResult<T> Runtime::issue_consensus_match(std::vector<T>&& input)
{
  return {std::move(input), legion_context_, legion_runtime_};
}

// ==========================================================================================

template <typename T>
T Runtime::get_tunable(Legion::MapperID mapper_id, int64_t tunable_id)
{
  return get_tunable(mapper_id, tunable_id, sizeof(T)).get_result<T>();
}

template <typename T>
T Runtime::get_core_tunable(int64_t tunable_id)
{
  return get_tunable<T>(core_library_->get_mapper_id(), tunable_id);
}

inline bool Runtime::initialized() const { return initialized_; }

inline const Library* Runtime::core_library() const { return core_library_; }

inline uint64_t Runtime::get_unique_store_id() { return next_store_id_++; }

inline uint64_t Runtime::get_unique_storage_id() { return next_storage_id_++; }

inline uint32_t Runtime::field_reuse_freq() const { return field_reuse_freq_; }

inline PartitionManager* Runtime::partition_manager() const { return partition_manager_; }

inline ProvenanceManager* Runtime::provenance_manager() const { return provenance_manager_; }

inline CommunicatorManager* Runtime::communicator_manager() const { return communicator_manager_; }

inline MachineManager* Runtime::machine_manager() const { return machine_manager_; }

}  // namespace legate::detail
