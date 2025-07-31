/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/attachment.h>
#include <legate/data/detail/external_allocation.h>
#include <legate/data/detail/region_field.h>
#include <legate/data/shape.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>

#include <legion.h>

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace legate::detail {

class Partition;
class Tiling;

/**
 * A `LogicalRegionField` is in essence a pair of a logical region and a field backing a
 * `LogicalStore` and its aliases created via store transformations. A `LogicalRegionField` also
 * stores its physical state in a `PhysicalState` object, in case it has been constructed from an
 * external allocation or has an inline mapping created for it.
 *
 * Because Legate tasks are scheduled in a deferred manner via scheduling window, any changes to the
 * physical state also needs to be deferred so that the tasks can see those changes in the right
 * order. The following summarizes how attaching, unmapping, and detaching are performed:
 *
 *  (1) At the time an `ExternalAllocation` is attached to a `LogicalRegionField`, the runtime sets
 *  `true` to `LogicalRegionField::attached_` to indicate that there's a pending
 *  `Attach`/`IndexAttach` operation for this `LogicalRegionField`. This bookkeeping allows the
 *  subsequent operations accessing this `LogicalRegionField` to know the pending attachment without
 *  examining the scheduling window. The issued operation then performs the actual attachment by
 *  updating `physical_state_.attachment_` of the `LogicalRegionField`.
 *
 *  (2) When a given `LogicalRegionField` loses all user references, the runtime frees its logical
 *  region and field and adds them to the list of free region fields, while the outstanding
 *  operations that use the region field are still sitting in the scheduling window.  The runtime
 *  safely performs the unmapping and detachment for the `LogicalRegionField` by issuing an
 *  `UnmapAndDetach` operation that unmaps the physical region and detaches the attachment only
 *  after all the preceding tasks using the `LogicalRegionField` are launched.
 *
 * Note that the inline mapping is performed immediately, as it always flushes the scheduling
 * window.
 */
class LogicalRegionField : public legate::EnableSharedFromThis<LogicalRegionField> {
 public:
  class PhysicalState {
   public:
    [[nodiscard]] const Legion::PhysicalRegion& ensure_mapping(const Legion::LogicalRegion& region,
                                                               Legion::FieldID field_id,
                                                               legate::mapping::StoreTarget target);
    void set_physical_region(Legion::PhysicalRegion physical_region);
    void set_attachment(Attachment attachment);
    void set_has_pending_detach(bool has_pending_detach);
    void add_callback(std::function<void()> callback);

    /**
     * @brief Remove all inline mappings (instances accessible by the top-level task) of this
     * LogicalRegionField.
     */
    void unmap();
    /**
     * @brief Detach this LogicalRegionField from any memory it's attached to.
     *
     * @param unordered Whether this operation is being invoked at a point in time that can differ
     * across the processes in a multi-process run, e.g. as part of garbage collection.
     */
    void detach(bool unordered);
    /**
     * @brief Ensure this LogicalRegionField is both unmapped and detached.
     *
     * @param unordered Whether this operation is being invoked at a point in time that can differ
     * across the processes in a multi-process run, e.g. as part of garbage collection.
     */
    void unmap_and_detach(bool unordered);
    void invoke_callbacks();
    void deallocate_attachment(bool wait_on_detach = true);

    [[nodiscard]] bool has_attachment() const;
    /**
     * @return Whether this logical region field has any invalidation callbacks.
     */
    [[nodiscard]] bool has_callbacks() const;

    [[nodiscard]] const Legion::PhysicalRegion& physical_region() const;
    [[nodiscard]] const Attachment& attachment() const;

    void intentionally_leak_physical_region();

   private:
    bool has_pending_detach_{};
    Legion::PhysicalRegion physical_region_{};
    Attachment attachment_{};
    std::vector<std::function<void()>> callbacks_{};
  };

  LogicalRegionField(InternalSharedPtr<Shape> shape,
                     std::uint32_t field_size,
                     Legion::LogicalRegion lr,
                     Legion::FieldID fid,
                     std::optional<InternalSharedPtr<LogicalRegionField>> parent = std::nullopt);

  ~LogicalRegionField() noexcept;

  [[nodiscard]] std::int32_t dim() const;
  [[nodiscard]] const Legion::LogicalRegion& region() const;
  [[nodiscard]] Legion::FieldID field_id() const;
  [[nodiscard]] const std::optional<InternalSharedPtr<LogicalRegionField>>& parent() const;
  [[nodiscard]] const LogicalRegionField& get_root() const;
  [[nodiscard]] Legion::Domain domain() const;
  [[nodiscard]] bool is_mapped() const;

  [[nodiscard]] RegionField map(legate::mapping::StoreTarget target);
  void unmap();
  void attach(Legion::PhysicalRegion physical_region,
              InternalSharedPtr<ExternalAllocation> allocation);
  void attach(Legion::ExternalResources external_resources,
              std::vector<InternalSharedPtr<ExternalAllocation>> allocations);
  void mark_pending_attach();
  void detach();
  void allow_out_of_order_destruction();
  void release_region_field() noexcept;

  [[nodiscard]] InternalSharedPtr<LogicalRegionField> get_child(const Tiling* tiling,
                                                                Span<const std::uint64_t> color,
                                                                bool complete);
  [[nodiscard]] Legion::LogicalPartition get_legion_partition(const Partition* partition,
                                                              bool complete);

  void add_invalidation_callback(std::function<void()> callback);
  void perform_invalidation_callbacks();

  // Should never copy or move raw logical region field objects
  LogicalRegionField(const LogicalRegionField&)            = delete;
  LogicalRegionField& operator=(const LogicalRegionField&) = delete;
  LogicalRegionField(LogicalRegionField&&)                 = delete;
  LogicalRegionField& operator=(LogicalRegionField&&)      = delete;

  [[nodiscard]] std::string to_string() const;

 private:
  InternalSharedPtr<Shape> shape_{};
  std::uint32_t field_size_{};
  Legion::LogicalRegion lr_{};
  Legion::FieldID fid_{};
  std::optional<InternalSharedPtr<LogicalRegionField>> parent_{};

  // These are flags that are updated immediately following the control flow
  bool released_{};
  bool mapped_{};
  bool attached_{};
  bool destroyed_out_of_order_{};

  // This object is updated in a deferred manner by an UnmapAndDetach operation
  InternalSharedPtr<PhysicalState> physical_state_{};
};

}  // namespace legate::detail

#include <legate/data/detail/logical_region_field.inl>
