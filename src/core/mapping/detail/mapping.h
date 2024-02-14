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

#include "core/mapping/detail/store.h"
#include "core/mapping/mapping.h"

#include <memory>
#include <set>
#include <vector>

namespace legate::mapping::detail {

TaskTarget to_target(Processor::Kind kind);

Processor::Kind to_kind(TaskTarget target);

Memory::Kind to_kind(StoreTarget target);

LegateVariantCode to_variant_code(TaskTarget target);

struct DimOrdering {
 public:
  using Kind = mapping::DimOrdering::Kind;

  explicit DimOrdering(Kind _kind);
  explicit DimOrdering(std::vector<std::int32_t> _dims);

  bool operator==(const DimOrdering& other) const;

  void populate_dimension_ordering(std::uint32_t ndim,
                                   std::vector<Legion::DimensionKind>& ordering) const;

  Kind kind;
  std::vector<std::int32_t> dims{};
};

struct StoreMapping {
 public:
  [[nodiscard]] bool for_future() const;
  [[nodiscard]] bool for_unbound_store() const;
  [[nodiscard]] const Store* store() const;

  [[nodiscard]] std::uint32_t requirement_index() const;
  [[nodiscard]] std::set<std::uint32_t> requirement_indices() const;
  [[nodiscard]] std::set<const Legion::RegionRequirement*> requirements() const;

  void populate_layout_constraints(Legion::LayoutConstraintSet& layout_constraints) const;

  [[nodiscard]] static std::unique_ptr<StoreMapping> default_mapping(const Store* store,
                                                                     StoreTarget target,
                                                                     bool exact = false);
  [[nodiscard]] static std::unique_ptr<StoreMapping> create(const Store* store,
                                                            InstanceMappingPolicy&& policy);

  std::vector<const Store*> stores{};
  InstanceMappingPolicy policy{};
};

}  // namespace legate::mapping::detail

#include "core/mapping/detail/mapping.inl"
