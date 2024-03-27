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

#include "legion.h"

#include <vector>

namespace legate::detail {

template <typename T>
class ConsensusMatchResult {
 private:
  friend class Runtime;
  ConsensusMatchResult(std::vector<T>&& input, Legion::Context ctx, Legion::Runtime* runtime);

 public:
  ~ConsensusMatchResult() noexcept;
  ConsensusMatchResult(ConsensusMatchResult&&) noexcept            = default;
  ConsensusMatchResult& operator=(ConsensusMatchResult&&) noexcept = default;

  ConsensusMatchResult(const ConsensusMatchResult&)            = delete;
  ConsensusMatchResult& operator=(const ConsensusMatchResult&) = delete;

  void wait();
  [[nodiscard]] const std::vector<T>& input() const;
  [[nodiscard]] const std::vector<T>& output() const;

 private:
  std::vector<T> input_{};
  std::vector<T> output_{};
  Legion::Future future_{};
  bool complete_{};
};

}  // namespace legate::detail

#include "core/runtime/detail/consensus_match_result.inl"
