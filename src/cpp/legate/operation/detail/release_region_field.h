/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_region_field.h>
#include <legate/operation/detail/operation.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <cstdint>

namespace legate::detail {

class ReleaseRegionField final : public Operation {
 public:
  /**
   * @brief An operation that cleans up the physical state of a freed region field in the pool. The
   * operation is issued such that when the region field is recycled for a downstream store
   * creation, the `ReleaseRegionField` op would run before any downstream consumers of the store
   * run.
   *
   * @param unique_id Unique ID for this operation.
   * @param physical_state Pointer to the released LogicalRegionField's physical status tracker.
   * @param unordered Whether this operation is being invoked at a point in time that can differ
   * across the processes in a multi-process run, e.g. as part of garbage collection.
   */
  ReleaseRegionField(std::uint64_t unique_id,
                     InternalSharedPtr<LogicalRegionField::PhysicalState> physical_state,
                     bool unordered);

  void launch() override;

  [[nodiscard]] Kind kind() const override;

  /**
   * @return `false`, `ReleaseRegionField` operations are inherently lazy, and never need to be
   * actively submitted.
   */
  [[nodiscard]] bool needs_flush() const override;

  /**
   * @return `false`, `ReleaseRegionField` operations operate on pure physical states, and
   * don't require the logical partitioning mechanisms.
   */
  [[nodiscard]] bool needs_partitioning() const override;

  /**
   * ReleaseRegionField operation is streamable if the RegionField it is
   * trying to release is still mapped or does not have any invalidation
   * callbacks. If it has either of them then the RegionField does not
   * need to be released. This operation needs to be assigned to unreleased
   * RegionField inside a Streaming Scope.
   *
   * @return Whether this operation supports streaming.
   */
  [[nodiscard]] bool supports_streaming() const override;

 private:
  InternalSharedPtr<LogicalRegionField::PhysicalState> physical_state_{};
  bool unordered_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/release_region_field.inl>
