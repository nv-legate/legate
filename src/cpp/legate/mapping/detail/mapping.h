/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/store.h>
#include <legate/mapping/mapping.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

namespace legate::mapping::detail {

[[nodiscard]] TaskTarget to_target(Processor::Kind kind);

[[nodiscard]] TaskTarget get_matching_task_target(StoreTarget target);

[[nodiscard]] StoreTarget to_target(Memory::Kind kind);

[[nodiscard]] Processor::Kind to_kind(TaskTarget target);

[[nodiscard]] Processor::Kind to_kind(VariantCode code);

[[nodiscard]] Memory::Kind to_kind(StoreTarget target);

[[nodiscard]] VariantCode to_variant_code(TaskTarget target);

[[nodiscard]] VariantCode to_variant_code(Processor::Kind kind);

class DimOrdering {
 public:
  using Kind = mapping::DimOrdering::Kind;

  explicit DimOrdering(Kind _kind);
  explicit DimOrdering(std::vector<std::int32_t> _dims);

  [[nodiscard]] bool operator==(const DimOrdering& other) const;

  void populate_dimension_ordering(std::uint32_t ndim,
                                   std::vector<Legion::DimensionKind>& ordering) const;

  Kind kind{};
  std::vector<std::int32_t> dims{};
};

class StoreMapping {
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

#include <legate/mapping/detail/mapping.inl>
