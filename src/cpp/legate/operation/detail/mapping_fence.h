/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/operation.h>

#include <cstdint>

namespace legate::detail {

class MappingFence final : public Operation {
 public:
  explicit MappingFence(std::uint64_t unique_id);

  void launch() override;

  [[nodiscard]] Kind kind() const override;
};

}  // namespace legate::detail

#include <legate/operation/detail/mapping_fence.inl>
