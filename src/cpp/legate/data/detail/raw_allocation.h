/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/raw_allocation.h>
#include <legate/mapping/mapping.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <cstddef>
#include <optional>

namespace legate::detail {

class RawAllocation {
 public:
  using Deleter = legate::RawAllocation::Deleter;

  RawAllocation(bool read_only,
                mapping::StoreTarget target,
                void* ptr,
                std::size_t size,
                std::optional<Deleter> deleter = std::nullopt);

  // The allocation owns its memory: when the last reference to it is dropped, its deleter (if any)
  // is run. This makes the raw allocation a self-owning, reference-counted handle that can be
  // shared freely by the storage, the physical store, and any inline-storage buffers viewing it.
  ~RawAllocation();

  [[nodiscard]] bool read_only() const;
  [[nodiscard]] mapping::StoreTarget target() const;
  [[nodiscard]] void* ptr() const;
  [[nodiscard]] std::size_t size() const;
  void maybe_deallocate();

  // detail::RawAllocation should never be copied or moved
  RawAllocation(const RawAllocation& other)            = delete;
  RawAllocation& operator=(const RawAllocation& other) = delete;
  RawAllocation(RawAllocation&& other)                 = delete;
  RawAllocation& operator=(RawAllocation&& other)      = delete;

  /**
   * @brief Allocates a fresh, Legate-owned raw allocation of @p size bytes on @p target.
   *
   * The returned allocation owns its memory and frees it (via the appropriate deleter for the
   * target memory kind) when its last reference is dropped. This is the shared allocation path
   * used by `Buffer`s (both storage-backing and task-local).
   *
   * @param target The memory kind to allocate on.
   * @param size The size of the allocation in bytes.
   *
   * @return The freshly allocated raw allocation.
   */
  [[nodiscard]] static InternalSharedPtr<RawAllocation> create(mapping::StoreTarget target,
                                                               std::size_t size);

 private:
  bool read_only_{};
  mapping::StoreTarget target_{mapping::StoreTarget::SYSMEM};
  void* ptr_{};
  std::size_t size_{};
  std::optional<Deleter> deleter_{};
};

}  // namespace legate::detail

#include <legate/data/detail/raw_allocation.inl>
