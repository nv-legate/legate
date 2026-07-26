/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/buffer.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

namespace legate::detail {

/**
 * @brief A data-movement request that copies the whole contents of a storage between two
 * `Buffer` allocations that may live on different memories (e.g. the GPU framebuffer and
 * system memory).
 *
 * Unlike `CopyRequest`, a remap request does not repartition data across ranks; it is a straight,
 * rank-local copy of the entire buffer. It is used to serve a task from a temporary redundant copy
 * on the memory the task runs on while keeping the canonical ("active") storage in place. A remap
 * copy-in populates the redundant copy from the active storage before the task runs; a remap
 * copy-out merges the redundant copy back into the active storage after the task runs.
 */
class RemapRequest {
 public:
  /**
   * @brief Constructs a remap request that copies the whole @p source buffer into @p target.
   *
   * @param source The storage to copy from.
   * @param target The storage to copy into. Must span the same domain as @p source.
   */
  RemapRequest(InternalSharedPtr<Buffer> source, InternalSharedPtr<Buffer> target);

  void perform(VariantCode variant_code) const;

  /**
   * @brief Whether performing this request requires a subsequent host synchronization.
   *
   * A device-to-host copy is issued asynchronously on the CUDA stream, so the host must wait for
   * its completion before reading the destination. Host-to-device and host-to-host copies do not
   * need this.
   *
   * @return `true` if a host synchronization is required after performing this request.
   */
  [[nodiscard]] bool requires_host_sync() const;

  RemapRequest(const RemapRequest&)            = default;
  RemapRequest& operator=(const RemapRequest&) = default;

  RemapRequest(RemapRequest&&) noexcept            = default;
  RemapRequest& operator=(RemapRequest&&) noexcept = default;

 private:
  InternalSharedPtr<Buffer> source_{};
  InternalSharedPtr<Buffer> target_{};
};

}  // namespace legate::detail

#include <legate/data/detail/remap_request.inl>
