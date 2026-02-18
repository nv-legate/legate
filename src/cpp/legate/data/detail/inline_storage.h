/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/external_allocation.h>
#include <legate/mapping/mapping.h>
#include <legate/utilities/typedefs.h>

#include <realm/instance.h>

#include <cstdint>
#include <utility>

namespace legate::mapping {

enum class StoreTarget : std::uint8_t;

}

namespace legate::detail {

class Type;

/**
 * @brief Class representing a manually-managed storage backend.
 */
class InlineStorage {
 public:
  /**
   * @brief Constructs an `InlineStorage` by allocating memory on the given target.
   *
   * Note: lifetime of allocated memory is tied to the lifetime of this object.
   *
   * @param domain The domain the storage should span.
   * @param field_size The size of the field in bytes.
   * @param target The target memory kind to allocate.
   */
  InlineStorage(const Domain& domain, std::uint32_t field_size, mapping::StoreTarget target);

  /**
   * @brief Construct an `InlineStorage` from an existing external allocation.
   *
   * Note: no guarantee on the lifetime of the allocated memory as it is given by caller.
   *
   * @param domain The domain the storage should span.
   * @param field_size The size of the field in bytes.
   * @param alloc The external allocation to use for the manually managed memory.
   */
  InlineStorage(const Domain& domain,
                std::uint32_t field_size,
                const legate::ExternalAllocation& alloc);

  InlineStorage(InlineStorage&&) noexcept                 = default;
  InlineStorage& operator=(InlineStorage&&) noexcept      = default;
  InlineStorage(const InlineStorage&) noexcept            = default;
  InlineStorage& operator=(const InlineStorage&) noexcept = default;
  ~InlineStorage();

  /**
   * @brief Returns the domain that the inline storage spans.
   */
  [[nodiscard]] const Domain& domain() const;

  /**
   * @brief Returns the memory kind of the inline storage.
   */
  [[nodiscard]] mapping::StoreTarget target() const;

  /**
   * @brief Returns a raw pointer to the memory allocation managed by the inline storage.
   */
  [[nodiscard]] void* data();

  /**
   * @brief Returns a const raw pointer to the memory allocation managed by the inline storage.
   */
  [[nodiscard]] const void* data() const;

  /**
   * @brief Returns the region instance wrapping the memory allocation of the inline storage.
   *
   * This is useful for information such as constructing accessors for interaction with
   * the underlying memory.
   *
   * @return The region instance wrapping the memory allocation of the inline storage.
   */
  [[nodiscard]] std::pair<Realm::RegionInstance, Realm::FieldID> region_instance() const;

  /**
   * @brief Remaps the current inline storage to a new target memory kind.
   *
   * This will free the currently existing memory allocation if remapping to a new target.
   *
   * @param new_target The new target memory kind to remap to.
   */
  void remap_to(mapping::StoreTarget new_target);

 private:
  [[nodiscard]] const legate::ExternalAllocation& alloc_() const;

  Domain domain_{};
  legate::ExternalAllocation allocation_{};
  Realm::RegionInstance region_instance_{};
};

}  // namespace legate::detail

#include <legate/data/detail/inline_storage.inl>
