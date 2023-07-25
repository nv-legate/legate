/* Copyright 2021-2022 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
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
  void populate_dimension_ordering(const Store* store,
                                   std::vector<Legion::DimensionKind>& ordering) const;

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
