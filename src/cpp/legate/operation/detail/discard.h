/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/operation.h>

#include <cstdint>

namespace legate::detail {

class Discard final : public Operation {
 public:
  Discard(std::uint64_t unique_id, Legion::LogicalRegion region, Legion::FieldID field_id);

  void launch() override;

  [[nodiscard]] Kind kind() const override;

 private:
  Legion::LogicalRegion region_{};
  Legion::FieldID field_id_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/discard.inl>
