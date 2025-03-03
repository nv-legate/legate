/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

 private:
  bool block_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/execution_fence.inl>
