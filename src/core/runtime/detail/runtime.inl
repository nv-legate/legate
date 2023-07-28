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
  : input_(std::move(input)),
    output_(input_.size()),
    future_(runtime->consensus_match(ctx, input_.data(), output_.data(), input_.size(), sizeof(T)))
{
}

template <typename T>
ConsensusMatchResult<T>::~ConsensusMatchResult()
{
  // Make sure the consensus match operation has completed, because it will be scribbling over the
  // buffers in this object.
  if (future_.valid()) wait();
}

template <typename T>
void ConsensusMatchResult<T>::wait()
{
  if (complete_) return;
  size_t num_matched = future_.get_result<size_t>();
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
  return ConsensusMatchResult<T>(std::move(input), legion_context_, legion_runtime_);
}

}  // namespace legate::detail
