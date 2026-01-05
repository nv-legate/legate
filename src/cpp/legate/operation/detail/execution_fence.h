/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/operation.h>

#include <cstdint>

namespace legate::detail {

class ExecutionFence final : public Operation {
 public:
  ExecutionFence(std::uint64_t unique_id, bool block);

  void launch() override;

  [[nodiscard]] Kind kind() const override;

  /**
   * @return `true` if the `ExecutionFence` is blocking, `false` otherwise.
   */
  [[nodiscard]] bool needs_flush() const override;

  /**
   * @return `false`, `ExecutionFence` operations operate on the scheduling window, not stores,
   * and are therefore partition-agnostic.
   */
  [[nodiscard]] bool needs_partitioning() const override;

 private:
  bool block_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/execution_fence.inl>
