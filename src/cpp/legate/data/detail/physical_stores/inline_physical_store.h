/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/inline_storage.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/shape.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

namespace legate::mapping {

enum class StoreTarget : std::uint8_t;

}

namespace legate::detail {

/**
 * @brief Physical store implementation for manually-managed memories.
 *
 * This PhysicalStore holds an InlineStorage that holds the manually-managed memory.
 */
class InlinePhysicalStore final : public PhysicalStore {
 public:
  /**
   * @brief Constructs an `InlinePhysicalStore`.
   *
   * @param dim The dimensionality of the store after transformations on the underlying storage.
   * @param type The type of the elements of the store.
   * @param redop_id The reduction operation ID.
   * @param transform The transformation stack of the physical store over the storage.
   * @param priv The privilege mode of this physical store.
   * @param storage The storage that holds the manually-managed memory.
   * @param domain The (sub)domain over the storage this physical store spans before transforms.
   */
  InlinePhysicalStore(std::int32_t dim,
                      InternalSharedPtr<Type> type,
                      GlobalRedopID redop_id,
                      InternalSharedPtr<detail::TransformStack> transform,
                      Legion::PrivilegeMode priv,
                      InternalSharedPtr<InlineStorage> storage,
                      const Domain& domain);

  /**
   * @brief Returns validity of the inline physical store.
   */
  [[nodiscard]] bool valid() const override;

  /**
   * @brief Returns the domain of the physical store.
   *
   * This accounts for the transformations on the provided (sub)domain of the underlying storage.
   *
   * @return The domain of the physical store.
   */
  [[nodiscard]] Domain domain() const override;

  /**
   * @brief Returns the inline allocation of the inline physical store.
   */
  [[nodiscard]] InlineAllocation get_inline_allocation() const override;

  /**
   * @brief Returns the target of the inline physical store.
   */
  [[nodiscard]] mapping::StoreTarget target() const override;

  /**
   * @brief Returns whether the inline physical store is partitioned.
   */
  [[nodiscard]] bool is_partitioned() const override;

  /**
   * @brief Returns the region instance and field ID wrapping the underlying memory allocation.
   *
   * This is useful for information such as constructing Legion accessors for interaction
   * with the underlying memory.
   *
   * @return std::pair holding the region instance and field ID wrapping wrapped memory allocation.
   */
  [[nodiscard]] std::pair<Realm::RegionInstance, Realm::FieldID> get_region_instance() const;

 private:
  [[nodiscard]] const InternalSharedPtr<InlineStorage>& storage_() const;

  // The inline storage that holds the manually-managed memory of the store.
  InternalSharedPtr<InlineStorage> inline_storage_{};

  // The domain over inline_storage_ that this store spans.
  std::reference_wrapper<const Domain> domain_;
};

}  // namespace legate::detail

#include <legate/data/detail/physical_stores/inline_physical_store.inl>
