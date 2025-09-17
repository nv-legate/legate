/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/mapping.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <tuple>

namespace legate::detail {

class TransformStack;

}  // namespace legate::detail

namespace legate::mapping::detail {

class RegionField {
 public:
  using Id = std::tuple<bool, std::uint32_t, Legion::FieldID>;

  RegionField() = default;
  RegionField(const Legion::RegionRequirement& req,
              std::int32_t dim,
              std::uint32_t idx,
              Legion::FieldID fid,
              bool unbound);

  [[nodiscard]] bool valid() const;

  [[nodiscard]] bool can_colocate_with(const RegionField& other) const;

  [[nodiscard]] Legion::Domain domain(Legion::Mapping::MapperRuntime& runtime,
                                      Legion::Mapping::MapperContext context) const;

  // REVIEW: this is not defined anywhere
  bool operator==(const RegionField& other) const;

  [[nodiscard]] Id unique_id() const;

  [[nodiscard]] std::int32_t dim() const;
  [[nodiscard]] std::uint32_t index() const;
  [[nodiscard]] Legion::FieldID field_id() const;
  [[nodiscard]] bool unbound() const;

  [[nodiscard]] const Legion::RegionRequirement& get_requirement() const;
  [[nodiscard]] Legion::IndexSpace get_index_space() const;

 private:
  // This is a pointer (not a reference_wrapper) because this class needs to be serialized,
  // which requires it to be default-constructible. But it is for all intents and purposes a
  // reference_wrapper.
  const Legion::RegionRequirement* req_{};
  std::int32_t dim_{-1};
  std::uint32_t idx_{-1U};
  Legion::FieldID fid_{-1U};
  bool unbound_{false};
};

class FutureWrapper {
 public:
  FutureWrapper() = default;
  FutureWrapper(std::uint32_t idx, const Legion::Domain& domain);

  [[nodiscard]] std::int32_t dim() const;
  [[nodiscard]] std::uint32_t index() const;
  [[nodiscard]] const Legion::Domain& domain() const;

 private:
  std::uint32_t idx_{-1U};
  Legion::Domain domain_{};
};

class Store {
 public:
  Store() = default;
  Store(std::int32_t dim,
        InternalSharedPtr<legate::detail::Type> type,
        FutureWrapper future,
        InternalSharedPtr<legate::detail::TransformStack>&& transform = nullptr);
  Store(Legion::Mapping::MapperRuntime& runtime,
        Legion::Mapping::MapperContext context,
        std::int32_t dim,
        InternalSharedPtr<legate::detail::Type> type,
        GlobalRedopID redop_id,
        const RegionField& region_field,
        bool is_unbound_store                                         = false,
        InternalSharedPtr<legate::detail::TransformStack>&& transform = nullptr);
  // A special constructor to create a mapper view of a store from a region requirement
  Store(Legion::Mapping::MapperRuntime& runtime,
        Legion::Mapping::MapperContext context,
        const Legion::RegionRequirement& requirement);

  [[nodiscard]] bool is_future() const;
  [[nodiscard]] bool unbound() const;
  [[nodiscard]] bool valid() const;
  [[nodiscard]] std::int32_t dim() const;
  [[nodiscard]] const InternalSharedPtr<legate::detail::Type>& type() const;
  /**
   * @brief Returns if the store is transformed
   *
   * @returns true if the store is transformed
   */
  [[nodiscard]] bool transformed() const;

  [[nodiscard]] bool is_reduction() const;
  [[nodiscard]] GlobalRedopID redop() const;

  [[nodiscard]] bool can_colocate_with(const Store& other) const;
  [[nodiscard]] const RegionField& region_field() const;
  [[nodiscard]] const FutureWrapper& future() const;

  [[nodiscard]] RegionField::Id unique_region_field_id() const;
  [[nodiscard]] std::uint32_t requirement_index() const;
  [[nodiscard]] std::uint32_t future_index() const;

  [[nodiscard]] Domain domain() const;

  [[nodiscard]] legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM> find_imaginary_dims()
    const;

  /**
   * @brief applies inverse transform on a tuple representing the dimensions of a
   * store.
   *
   * @param dims a tuple of integer dimensions with values in the
   *        range [0..dim()).
   *
   * @returns the transformed tuple of dimensions.
   */
  [[nodiscard]] legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM> invert_dims(
    legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM> dims) const;

 private:
  bool is_future_{};
  bool is_unbound_store_{};
  std::int32_t dim_{-1};
  InternalSharedPtr<legate::detail::Type> type_{};
  GlobalRedopID redop_id_{-1};

  FutureWrapper future_{};
  RegionField region_field_{};

  InternalSharedPtr<legate::detail::TransformStack> transform_{};

  Legion::Mapping::MapperRuntime* runtime_{};
  Legion::Mapping::MapperContext context_{};
};

}  // namespace legate::mapping::detail

#include <legate/mapping/detail/store.inl>
