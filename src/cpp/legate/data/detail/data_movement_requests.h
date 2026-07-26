/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/buffer.h>
#include <legate/data/detail/copy_request.h>
#include <legate/data/detail/reduction_request.h>
#include <legate/data/detail/remap_request.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace legate::detail {

class DataMovementRequests {
 public:
  void register_copy_in(CopyRequest&& copy_in);
  void register_copy_out(CopyRequest&& copy_out);
  void register_reduction(ReductionRequest&& reduction);

  // Remap requests migrate a storage between memories (e.g. GPU framebuffer and system memory)
  // while keeping the canonical ("active") storage in place. They are ordered around the
  // repartitioning copies: remap copy-ins run before repartitioning copy-ins so a redundant copy is
  // populated before it is redistributed, and remap copy-outs run after repartitioning copy-outs so
  // the redundant copy is fully merged back before it is migrated to the active storage.
  void register_remap_copy_in(RemapRequest&& copy_in);
  void register_remap_copy_out(RemapRequest&& copy_out);

  void require_data_sync();

  void reserve_copy_ins(std::size_t size);
  void reserve_copy_outs(std::size_t size);
  void reserve_reduction_requests(std::size_t size);

  void perform_data_sync() const;
  void perform_remap_copy_ins(VariantCode variant_code) const;
  void perform_copy_ins(VariantCode variant_code) const;
  void perform_reductions(VariantCode variant_code) const;
  void perform_copy_outs(VariantCode variant_code) const;
  void perform_remap_copy_outs(VariantCode variant_code) const;

  /**
   * @brief Looks up a physical allocation already handed out for a given storage during the
   * current task launch.
   *
   * This is used to make a store passed as both an input and an output (or otherwise accessed
   * multiple times with the same partition within a single task) map to a single physical
   * allocation, so that updates made by the task are seen in place rather than being lost in a
   * second, independent allocation.
   *
   * @param storage_id The id of the storage being mapped.
   *
   * @return The cached allocation, or `std::nullopt` if none has been recorded yet.
   */
  [[nodiscard]] std::optional<InternalSharedPtr<Buffer>> find_mapped_alloc(
    std::uint64_t storage_id) const;
  void record_mapped_alloc(std::uint64_t storage_id, InternalSharedPtr<Buffer> alloc);

  /**
   * @brief Looks up the redundant "working copy" of a storage created for the current task launch.
   *
   * When a task runs on a different memory than a storage's active (canonical) `Buffer`, the
   * task operates on a redundant copy on its own memory instead. This copy is shared across the
   * input and output mappings of the same store so that a single remap copy-in populates it and a
   * single remap copy-out merges it back.
   *
   * @param storage_id The id of the storage being mapped.
   *
   * @return The cached working copy, or `std::nullopt` if none has been recorded yet.
   */
  [[nodiscard]] std::optional<InternalSharedPtr<Buffer>> find_working_copy(
    std::uint64_t storage_id) const;
  void record_working_copy(std::uint64_t storage_id, InternalSharedPtr<Buffer> alloc);

  bool empty() const;

 private:
  bool data_sync_required_{};
  std::vector<RemapRequest> remap_copy_ins_{};
  std::vector<CopyRequest> copy_ins_{};
  std::vector<CopyRequest> copy_outs_{};
  std::vector<RemapRequest> remap_copy_outs_{};
  std::vector<ReductionRequest> reductions_{};
  std::unordered_map<std::uint64_t, InternalSharedPtr<Buffer>> mapped_allocs_{};
  std::unordered_map<std::uint64_t, InternalSharedPtr<Buffer>> working_copies_{};
};

}  // namespace legate::detail

#include <legate/data/detail/data_movement_requests.inl>
