/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

  /**
   * @brief Create a Legion region requirement from this projection store.
   *
   * @param region The logical region for which to create the requirement.
   * @param fields The fields of `region` for which the requirement should apply.
   * @param privilege The privilege mode for the region.
   * @param is_key Whether the region is the "key" region.
   * @param is_single Whether the region requires exclusive or collective exclusive coherence.
   *
   * @return The region requirement.
   */
  template <bool SINGLE>
  [[nodiscard]] Legion::RegionRequirement create_requirement(
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
  using BaseStoreProjection::create_requirement;

  /**
   * @brief Create a Legion region requirement from this projection store.
   *
   * @param region The logical region for which to create the requirement.
   * @param fields The fields of `region` for which the requirement should apply.
   * @param privilege The privilege mode for the region.
   *
   * @return The region requirement.
   */
  template <bool SINGLE>
  [[nodiscard]] Legion::RegionRequirement create_requirement(
    const Legion::LogicalRegion& region,
    const std::vector<Legion::FieldID>& fields,
    Legion::PrivilegeMode privilege) const;

  bool is_key{};
};

}  // namespace legate::detail

#include <legate/operation/detail/store_projection.inl>
