/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/hash.h>
#include <legate/utilities/typedefs.h>

namespace legate::detail {

class BaseStoreProjection {
 public:
  BaseStoreProjection() = default;

  BaseStoreProjection(Legion::LogicalPartition partition, Legion::ProjectionID proj_id);

  bool operator<(const BaseStoreProjection& other) const;
  bool operator==(const BaseStoreProjection& other) const;

  // TODO(wonchanl): Ideally we want this method to return a requirement, instead of taking an
  // inout argument. We go with an inout parameter for now, as RegionRequirement doesn't have a
  // move constructor/assignment.
  template <bool SINGLE>
  void populate_requirement(Legion::RegionRequirement& requirement,
                            const Legion::LogicalRegion& region,
                            const std::vector<Legion::FieldID>& fields,
                            Legion::PrivilegeMode privilege,
                            bool is_key,
                            bool is_single = SINGLE) const;

  void set_reduction_op(GlobalRedopID _redop);

  [[nodiscard]] std::size_t hash() const noexcept;

  Legion::LogicalPartition partition{Legion::LogicalPartition::NO_PART};
  Legion::ProjectionID proj_id{};
  GlobalRedopID redop{-1};
};

class StoreProjection final : public BaseStoreProjection {
 public:
  using BaseStoreProjection::BaseStoreProjection;
  using BaseStoreProjection::populate_requirement;

  template <bool SINGLE>
  void populate_requirement(Legion::RegionRequirement& requirement,
                            const Legion::LogicalRegion& region,
                            const std::vector<Legion::FieldID>& fields,
                            Legion::PrivilegeMode privilege) const;

  bool is_key{};
};

}  // namespace legate::detail

#include <legate/operation/detail/store_projection.inl>
