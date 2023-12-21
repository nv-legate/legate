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

#include "core/type/detail/type_info.h"
#include "core/utilities/internal_shared_ptr.h"
#include "core/utilities/typedefs.h"

#include <tuple>
#include <vector>

namespace legate::detail {
struct TransformStack;
}  // namespace legate::detail

namespace legate::mapping::detail {

class RegionField {
 public:
  using Id = std::tuple<bool, uint32_t, Legion::FieldID>;

  RegionField() = default;
  RegionField(const Legion::RegionRequirement* req, int32_t dim, uint32_t idx, Legion::FieldID fid);

  [[nodiscard]] bool can_colocate_with(const RegionField& other) const;

  [[nodiscard]] Legion::Domain domain(Legion::Mapping::MapperRuntime* runtime,
                                      Legion::Mapping::MapperContext context) const;

  // REVIEW: this is not defined anywhere
  bool operator==(const RegionField& other) const;

  [[nodiscard]] Id unique_id() const;

  [[nodiscard]] int32_t dim() const;
  [[nodiscard]] uint32_t index() const;
  [[nodiscard]] Legion::FieldID field_id() const;
  [[nodiscard]] bool unbound() const;

  [[nodiscard]] const Legion::RegionRequirement* get_requirement() const;
  [[nodiscard]] Legion::IndexSpace get_index_space() const;

 private:
  const Legion::RegionRequirement* req_{};
  int32_t dim_{-1};
  uint32_t idx_{-1U};
  Legion::FieldID fid_{-1U};
};

class FutureWrapper {
 public:
  FutureWrapper() = default;
  FutureWrapper(uint32_t idx, const Legion::Domain& domain);

  [[nodiscard]] int32_t dim() const;
  [[nodiscard]] uint32_t index() const;
  [[nodiscard]] Legion::Domain domain() const;

 private:
  uint32_t idx_{-1U};
  Legion::Domain domain_{};
};

class Store {
 public:
  Store() = default;
  Store(int32_t dim,
        InternalSharedPtr<legate::detail::Type> type,
        FutureWrapper future,
        InternalSharedPtr<legate::detail::TransformStack>&& transform = nullptr);
  Store(Legion::Mapping::MapperRuntime* runtime,
        Legion::Mapping::MapperContext context,
        int32_t dim,
        InternalSharedPtr<legate::detail::Type> type,
        int32_t redop_id,
        const RegionField& region_field,
        bool is_unbound_store                                         = false,
        InternalSharedPtr<legate::detail::TransformStack>&& transform = nullptr);
  // A special constructor to create a mapper view of a store from a region requirement
  Store(Legion::Mapping::MapperRuntime* runtime,
        Legion::Mapping::MapperContext context,
        const Legion::RegionRequirement* requirement);

  [[nodiscard]] bool is_future() const;
  [[nodiscard]] bool unbound() const;
  [[nodiscard]] int32_t dim() const;
  [[nodiscard]] InternalSharedPtr<legate::detail::Type> type() const;

  [[nodiscard]] bool is_reduction() const;
  [[nodiscard]] int32_t redop() const;

  [[nodiscard]] bool can_colocate_with(const Store& other) const;
  [[nodiscard]] const RegionField& region_field() const;
  [[nodiscard]] const FutureWrapper& future() const;

  [[nodiscard]] RegionField::Id unique_region_field_id() const;
  [[nodiscard]] uint32_t requirement_index() const;
  [[nodiscard]] uint32_t future_index() const;

  [[nodiscard]] Domain domain() const;

  [[nodiscard]] std::vector<int32_t> find_imaginary_dims() const;

 private:
  bool is_future_{};
  bool is_unbound_store_{};
  int32_t dim_{-1};
  InternalSharedPtr<legate::detail::Type> type_{};
  int32_t redop_id_{-1};

  FutureWrapper future_;
  RegionField region_field_;

  InternalSharedPtr<legate::detail::TransformStack> transform_{};

  Legion::Mapping::MapperRuntime* runtime_{};
  Legion::Mapping::MapperContext context_{};
};

}  // namespace legate::mapping::detail

#include "core/mapping/detail/store.inl"
