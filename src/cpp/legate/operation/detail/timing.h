/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/operation.h>
#include <legate/utilities/macros.h>

#include <cstdint>

namespace legate::detail {

class Strategy;

class Timing final : public Operation {
 public:
  enum class Precision : std::uint8_t {
    MICRO,
    NANO,
  };

  Timing(std::uint64_t unique_id, Precision precision, InternalSharedPtr<LogicalStore> store);

  void launch() override;
  // This method never gets invoked, but we define it just to appease compilers that don't like
  // partially overridden overloads (e.g., NVCC)
  void launch(Strategy* strategy) override;

  [[nodiscard]] Kind kind() const override;

  /**
   * @return `false`, `Timing` operations are inherently lazy (they time the actual task
   * bodies, not any launching overhead), and therefore don't need to be actively submitted.
   */
  [[nodiscard]] bool needs_flush() const override;

  /**
   * @return `false`, `Timing` operations are meta-operations that operate "on" task bodies
   * themselves, not any stores. So they don't need to be partitioned.
   */
  [[nodiscard]] bool needs_partitioning() const override;

 private:
  Precision precision_{Precision::MICRO};
  InternalSharedPtr<LogicalStore> store_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/timing.inl>
