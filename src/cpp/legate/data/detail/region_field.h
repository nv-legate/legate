/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/inline_allocation.h>
#include <legate/mapping/mapping.h>

#include <cstdint>
#include <optional>

namespace legate::detail {

class RegionField {
 public:
  RegionField() = default;

  /** @brief Constructor for a region field.
   *
   * This constructor assumes the domain of this region field is the same
   * as that of the logical region which could be a subset of the domain
   * of the physical region.  However, the physical region provides
   * the actual memory allocation for the region field.
   *
   * @param dim The dimension of the region field.
   * @param lr The logical region of the region field (could be a subset of the physical region).
   * @param pr The physical region of the region field.
   * @param fid The field ID of the region field.
   * @param partitioned Whether the region field is partitioned.
   */
  RegionField(std::int32_t dim,
              Legion::LogicalRegion lr,
              Legion::PhysicalRegion pr,
              Legion::FieldID fid,
              bool partitioned);

  /** @brief Constructor for a region field.
   *
   * This constructor assumes the domain of this region field is the same
   * as the domain of the entire physical region.
   *
   * @param dim The dimension of the region field.
   * @param pr The physical region of the region field.
   * @param fid The field ID of the region field.
   * @param partitioned Whether the region field is partitioned.
   */
  RegionField(std::int32_t dim, Legion::PhysicalRegion pr, Legion::FieldID fid, bool partitioned);

  RegionField(RegionField&& other) noexcept            = default;
  RegionField& operator=(RegionField&& other) noexcept = default;

  RegionField(const RegionField& other)            = delete;
  RegionField& operator=(const RegionField& other) = delete;

  [[nodiscard]] bool valid() const;

  [[nodiscard]] std::int32_t dim() const;

  [[nodiscard]] Domain domain() const;
  void set_logical_region(const Legion::LogicalRegion& lr);
  [[nodiscard]] InlineAllocation get_inline_allocation() const;
  [[nodiscard]] InlineAllocation get_inline_allocation(
    const Domain& domain, const Legion::DomainAffineTransform& transform) const;
  [[nodiscard]] mapping::StoreTarget target() const;

  [[nodiscard]] bool is_partitioned() const;
  [[nodiscard]] bool is_readable() const;
  [[nodiscard]] bool is_writable() const;
  [[nodiscard]] bool is_reducible() const;

  [[nodiscard]] const Legion::PhysicalRegion& get_physical_region() const;
  [[nodiscard]] Legion::FieldID get_field_id() const;
  [[nodiscard]] Legion::LogicalRegion get_logical_region() const;

 private:
  std::int32_t dim_{-1};
  std::optional<Legion::PhysicalRegion> pr_{};
  Legion::LogicalRegion lr_{};
  Legion::FieldID fid_{-1U};

  bool partitioned_{};
  bool readable_{};
  bool writable_{};
  bool reducible_{};
};

}  // namespace legate::detail

#include <legate/data/detail/region_field.inl>
