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

#include <memory>

#include "core/mapping/detail/store.h"
#include "core/mapping/mapping.h"

namespace legate::mapping::detail {

TaskTarget to_target(Processor::Kind kind);

Processor::Kind to_kind(TaskTarget target);

Memory::Kind to_kind(StoreTarget target);

LegateVariantCode to_variant_code(TaskTarget target);

struct DimOrdering {
 public:
  using Kind = mapping::DimOrdering::Kind;

 public:
  DimOrdering(Kind _kind) : kind(_kind) {}
  DimOrdering(const std::vector<int32_t>& _dims) : kind(Kind::CUSTOM), dims(_dims) {}

 public:
  DimOrdering(const DimOrdering&) = default;

 public:
  bool operator==(const DimOrdering& other) const
  {
    return kind == other.kind && dims == other.dims;
  }

 public:
  void populate_dimension_ordering(int32_t dim, std::vector<Legion::DimensionKind>& ordering) const;

 public:
  Kind kind;
  std::vector<int32_t> dims{};
};

struct StoreMapping {
 public:
  std::vector<const Store*> stores{};
  InstanceMappingPolicy policy;

 public:
  StoreMapping() {}

 public:
  StoreMapping(const StoreMapping&)            = default;
  StoreMapping& operator=(const StoreMapping&) = default;

 public:
  StoreMapping(StoreMapping&&)            = default;
  StoreMapping& operator=(StoreMapping&&) = default;

 public:
  bool for_future() const;
  bool for_unbound_store() const;
  const Store* store() const;

 public:
  uint32_t requirement_index() const;
  std::set<uint32_t> requirement_indices() const;
  std::set<const Legion::RegionRequirement*> requirements() const;

 public:
  void populate_layout_constraints(Legion::LayoutConstraintSet& layout_constraints) const;

 public:
  static std::unique_ptr<StoreMapping> default_mapping(const Store* store,
                                                       StoreTarget target,
                                                       bool exact = false);
  static std::unique_ptr<StoreMapping> create(const Store* store, InstanceMappingPolicy&& policy);
};

}  // namespace legate::mapping::detail
