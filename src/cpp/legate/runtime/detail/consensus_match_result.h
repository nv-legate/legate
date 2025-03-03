/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legion.h>

#include <vector>

namespace legate::detail {

template <typename T>
class ConsensusMatchResult {
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

#include <legate/runtime/detail/consensus_match_result.inl>
