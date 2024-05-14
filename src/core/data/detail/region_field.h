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

#include "core/data/inline_allocation.h"
#include "core/mapping/mapping.h"

#include <cstdint>
#include <optional>

namespace legate::detail {

class RegionField {
 public:
  RegionField() = default;
  RegionField(std::int32_t dim, Legion::PhysicalRegion pr, Legion::FieldID fid);

  RegionField(RegionField&& other) noexcept            = default;
  RegionField& operator=(RegionField&& other) noexcept = default;

  RegionField(const RegionField& other)            = delete;
  RegionField& operator=(const RegionField& other) = delete;

  [[nodiscard]] bool valid() const;

  [[nodiscard]] std::int32_t dim() const;

  [[nodiscard]] Domain domain() const;
  void set_logical_region(const Legion::LogicalRegion& lr);
  [[nodiscard]] InlineAllocation get_inline_allocation(std::uint32_t field_size) const;
  [[nodiscard]] InlineAllocation get_inline_allocation(
    std::uint32_t field_size,
    const Domain& domain,
    const Legion::DomainAffineTransform& transform) const;
  [[nodiscard]] mapping::StoreTarget target() const;

  [[nodiscard]] bool is_readable() const;
  [[nodiscard]] bool is_writable() const;
  [[nodiscard]] bool is_reducible() const;

  [[nodiscard]] const Legion::PhysicalRegion& get_physical_region() const;
  [[nodiscard]] Legion::FieldID get_field_id() const;

 private:
  std::int32_t dim_{-1};
  std::optional<Legion::PhysicalRegion> pr_{};
  Legion::LogicalRegion lr_{};
  Legion::FieldID fid_{-1U};

  bool readable_{};
  bool writable_{};
  bool reducible_{};
};

}  // namespace legate::detail

#include "core/data/detail/region_field.inl"
