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
#include "core/runtime/detail/consensus_match_result.h"
#include "core/utilities/assert.h"

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
  if (!future_.valid()) {
    return;
  }
  try {
    wait();
  } catch (const std::exception& excn) {
    LEGATE_ABORT(excn.what());
  }
}

template <typename T>
void ConsensusMatchResult<T>::wait()
{
  if (complete_) {
    return;
  }
  const auto num_matched = future_.get_result<std::size_t>();
  LegateCheck(num_matched <= output_.size());
  output_.resize(num_matched);
  complete_ = true;
}

template <typename T>
const std::vector<T>& ConsensusMatchResult<T>::input() const
{
  return input_;
}

template <typename T>
const std::vector<T>& ConsensusMatchResult<T>::output() const
{
  LegateCheck(complete_);
  return output_;
}

}  // namespace legate::detail
