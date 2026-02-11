/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/mapping/mapping.h>

#include <optional>

namespace legate::mapping {

/**
 * @brief An object that models the InstanceMappingPolicy class for use in specifying mapping
 * decisions in a `TaskConfig` object.
 *
 * This class is nearly identical to InstanceMappingPolicy except that the `target` member is a
 * `std::optional` instead of a `StoreTarget`. Setting `target` to `std::nullopt` indicates
 * that the `InstanceMappingPolicy` that is synthesized during mapping-time should take the
 * first available memory type from the given options. If `target` is set to a concrete
 * `StoreTarget`, then this store target is always chosen.
 */
class LEGATE_EXPORT ProxyInstanceMappingPolicy {
 public:
  std::optional<StoreTarget> target{};
  AllocPolicy allocation{AllocPolicy::MAY_ALLOC};
  std::optional<DimOrdering> ordering{};
  bool exact{};
  bool redundant{};

  /**
   * @brief Changes the store target
   *
   * @param tgt A new store target or `std::nullopt` if the target should be chosen at
   * mapping-time.
   *
   * @return This instance mapping policy
   */
  [[nodiscard]] ProxyInstanceMappingPolicy&& with_target(std::optional<StoreTarget> tgt) &&;

  /**
   * @brief Changes the allocation policy
   *
   * @param alloc A new allocation policy
   *
   * @return This instance mapping policy
   */
  [[nodiscard]] ProxyInstanceMappingPolicy&& with_allocation_policy(AllocPolicy alloc) &&;

  /**
   * @brief Changes the dimension ordering
   *
   * @param ord A new dimension ordering, or `std::nullopt` if the dimension ordering
   * should be chosen at mapping-time. See `InstanceMappingPolicy::ordering` for more
   * information.
   *
   * @return This instance mapping policy
   */
  [[nodiscard]] ProxyInstanceMappingPolicy&& with_ordering(std::optional<DimOrdering> ord) &&;

  /**
   * @brief Changes the value of `exact`
   *
   * @param value A new value for the `exact` field
   *
   * @return This instance mapping policy
   */
  [[nodiscard]] ProxyInstanceMappingPolicy&& with_exact(bool value) &&;

  /**
   * @brief Changes the value of `redundant`
   *
   * @param value A new value for the `redundant` field
   *
   * @return This instance mapping policy
   */
  [[nodiscard]] ProxyInstanceMappingPolicy&& with_redundant(bool value) &&;

  friend bool operator==(const ProxyInstanceMappingPolicy& lhs,
                         const ProxyInstanceMappingPolicy& rhs);
  friend bool operator!=(const ProxyInstanceMappingPolicy& lhs,
                         const ProxyInstanceMappingPolicy& rhs);
};

}  // namespace legate::mapping

#include <legate/mapping/proxy_instance_mapping_policy.inl>
