/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/external_allocation.h>
#include <legate/mapping/mapping.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/typedefs.h>

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
   * @brief Re-interpret the same allocation as spanning a different domain of equal volume.
   *
   * The affine layout (base pointer and strides) is derived purely from the domain, so changing
   * the domain re-bases the storage without touching the underlying memory. This is used to
   * relocate a zero-based buffer bound to an unbound store into its global sub-domain (offset)
   * within the fully-formed store, so the copy engine sees the same global-coordinate convention
   * it uses for tiled storages.
   *
   * @param domain The new domain. Must have the same volume as the current one.
   */
  void rebase(const Domain& domain);

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
   * @brief Returns the base pointer and per-dimension byte strides of the storage's affine
   * layout.
   *
   * The base pointer is the address of the element at the storage's coordinate origin (i.e. the
   * point `(0, ..., 0)`), following the affine convention `addr(p) = base + sum(p[i] *
   * strides[i])`. The strides are given in bytes and have as many entries as the storage
   * domain's dimension (at least one).
   *
   * @return A pair of the base pointer and the byte strides.
   */
  [[nodiscard]] std::pair<void*, SmallVector<std::uint64_t, LEGATE_MAX_DIM>> base_and_byte_strides()
    const;

 private:
  [[nodiscard]] const legate::ExternalAllocation& alloc_() const;

  Domain domain_{};
  std::uint32_t field_size_{};
  legate::ExternalAllocation allocation_{};
};

}  // namespace legate::detail

#include <legate/data/detail/inline_storage.inl>
