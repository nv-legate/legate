/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/store.h>
#include <legate/mapping/mapping.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

namespace legate::mapping::detail {

/**
 * @brief Convert a VariantCode to its corresponding TaskTarget.
 *
 * @param code The variant code to convert.
 *
 * @return The TaskTarget associated with the given VariantCode.
 */
[[nodiscard]] TaskTarget to_target(VariantCode code);

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

  /**
   * @brief Generates a vector of Legion dimensions of size `ndim`.
   *
   * @param ndim The number of dimensions to generate.
   *
   * @returns a vector of Legion dimensions.
   */
  [[nodiscard]] std::vector<Legion::DimensionKind> generate_legion_dims(std::uint32_t ndim) const;

  /**
   * @brief Generates a vector of Legion dimensions for a given Store
   *
   * @param store The store for which the Legion dimensions is created
   *
   * @returns a vector of Legion dimensions.
   */
  [[nodiscard]] std::vector<Legion::DimensionKind> generate_legion_dims(const Store& store) const;

  Kind kind{};
  std::vector<std::int32_t> dims{};
};

class StoreMapping {
 public:
  StoreMapping() = default;
  StoreMapping(InstanceMappingPolicy policy, const Store* store);
  StoreMapping(InstanceMappingPolicy policy, Span<const Store* const> stores);
  StoreMapping(InstanceMappingPolicy policy, Span<const InternalSharedPtr<Store>> stores);

  [[nodiscard]] Span<const Store* const> stores() const;
  [[nodiscard]] InstanceMappingPolicy& policy();
  [[nodiscard]] const InstanceMappingPolicy& policy() const;

  void add_store(const Store* store);

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

 private:
  InstanceMappingPolicy policy_{};
  std::vector<const Store*> stores_{};
};

}  // namespace legate::mapping::detail

#include <legate/mapping/detail/mapping.inl>
