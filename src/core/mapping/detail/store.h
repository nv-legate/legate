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
#include "core/utilities/typedefs.h"

namespace legate::detail {
struct TransformStack;
}  // namespace legate::detail

namespace legate::mapping::detail {

class RegionField {
 public:
  using Id = std::tuple<bool, uint32_t, Legion::FieldID>;

 public:
  RegionField() {}
  RegionField(const Legion::RegionRequirement* req, int32_t dim, uint32_t idx, Legion::FieldID fid);

 public:
  RegionField(const RegionField& other)            = default;
  RegionField& operator=(const RegionField& other) = default;

 public:
  bool can_colocate_with(const RegionField& other) const;

 public:
  Legion::Domain domain(Legion::Mapping::MapperRuntime* runtime,
                        const Legion::Mapping::MapperContext context) const;

 public:
  bool operator==(const RegionField& other) const;

 public:
  Id unique_id() const { return std::make_tuple(unbound(), idx_, fid_); }

 public:
  int32_t dim() const { return dim_; }
  uint32_t index() const { return idx_; }
  Legion::FieldID field_id() const { return fid_; }
  bool unbound() const { return dim_ < 0; }

 public:
  const Legion::RegionRequirement* get_requirement() const { return req_; }
  Legion::IndexSpace get_index_space() const;

 private:
  const Legion::RegionRequirement* req_{nullptr};
  int32_t dim_{-1};
  uint32_t idx_{-1U};
  Legion::FieldID fid_{-1U};
};

class FutureWrapper {
 public:
  FutureWrapper() {}
  FutureWrapper(uint32_t idx, const Legion::Domain& domain);

 public:
  FutureWrapper(const FutureWrapper& other)            = default;
  FutureWrapper& operator=(const FutureWrapper& other) = default;

 public:
  int32_t dim() const { return domain_.dim; }
  uint32_t index() const { return idx_; }

 public:
  Legion::Domain domain() const;

 private:
  uint32_t idx_{-1U};
  Legion::Domain domain_{};
};

class Store {
 public:
  Store() {}
  Store(int32_t dim,
        std::shared_ptr<legate::detail::Type> type,
        FutureWrapper future,
        std::shared_ptr<legate::detail::TransformStack>&& transform = nullptr);
  Store(Legion::Mapping::MapperRuntime* runtime,
        const Legion::Mapping::MapperContext context,
        int32_t dim,
        std::shared_ptr<legate::detail::Type> type,
        int32_t redop_id,
        const RegionField& region_field,
        bool is_unbound_store                                       = false,
        std::shared_ptr<legate::detail::TransformStack>&& transform = nullptr);
  // A special constructor to create a mapper view of a store from a region requirement
  Store(Legion::Mapping::MapperRuntime* runtime,
        const Legion::Mapping::MapperContext context,
        const Legion::RegionRequirement* requirement);

 public:
  Store(const Store& other)            = default;
  Store& operator=(const Store& other) = default;

 public:
  Store(Store&& other)            = default;
  Store& operator=(Store&& other) = default;

 public:
  bool is_future() const { return is_future_; }
  bool unbound() const { return is_unbound_store_; }
  int32_t dim() const { return dim_; }
  std::shared_ptr<legate::detail::Type> type() const { return type_; }

 public:
  bool is_reduction() const { return redop_id_ > 0; }
  int32_t redop() const { return redop_id_; }

 public:
  bool can_colocate_with(const Store& other) const;
  const RegionField& region_field() const;
  const FutureWrapper& future() const;

 public:
  RegionField::Id unique_region_field_id() const;
  uint32_t requirement_index() const;
  uint32_t future_index() const;

 public:
  Domain domain() const;

 public:
  std::vector<int32_t> find_imaginary_dims() const;

 private:
  bool is_future_{false};
  bool is_unbound_store_{false};
  int32_t dim_{-1};
  std::shared_ptr<legate::detail::Type> type_{};
  int32_t redop_id_{-1};

 private:
  FutureWrapper future_;
  RegionField region_field_;

 private:
  std::shared_ptr<legate::detail::TransformStack> transform_{nullptr};

 private:
  Legion::Mapping::MapperRuntime* runtime_{nullptr};
  Legion::Mapping::MapperContext context_{nullptr};
};

}  // namespace legate::mapping::detail
