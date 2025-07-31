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

  /**
   * Discards are always lazily applied, and their results are not externally visible.
   *
   * @return `false`, as `Discard` operations never need to immediately flush the scheduling
   * window.
   */
  [[nodiscard]] bool needs_flush() const override;

  /**
   * @return `false`, as `Discard` operations are performed on an entire LogicalRegion and
   * hence requires no partitioning.
   */
  [[nodiscard]] bool needs_partitioning() const override;

  /**
   * Discard operations are always streamable.
   *
   * @return Whether this operation supports streaming.
   */
  [[nodiscard]] bool supports_streaming() const override;

  /**
   * @return The `LogicalRegion` that this discard will apply to.
   */
  [[nodiscard]] const Legion::LogicalRegion& region() const;

  /**
   * @return The specific field in the `LogicalRegion` that this discard will apply to.
   */
  [[nodiscard]] Legion::FieldID field_id() const;

 private:
  Legion::LogicalRegion region_{};
  Legion::FieldID field_id_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/discard.inl>
