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
 * A `LogicalRegionField` is a pair of a logical region and a field backing a `LogicalStore` and its
 * aliases created via store transformations. A `LogicalRegionField` stays alive and occupies the
 * region field as long as there exists a `LogicalStore` associated with it. When a
 * `LogicalRegionField` loses all referring stores and gets destroyed, the region field that it has
 * occupied is returned to the runtime and recycled for another `LogicalStore` of the same shape and
 * element size.
 *
 * In addition to the logical region field, a `LogicalRegionField` keeps track of the region field's
 * physical state (`LogicalRegionField::PhysicalState`) in case the region field is ever
 * materialized in a physical allocation. The region field can be materialized by either inline
 * mapping (`LogicalRegionField::map`) or attachment (`LogicalRegionField::attach`). In both cases,
 * the runtime needs to make sure that the materialized allocation is destroyed when the
 * `LogicalRegionField` is destroyed and returned to the runtime's pool of region fields. The
 * attachment case needs a special treatment, as the `attach()` call is made by a deferred operation
 * (`Attach`) and not eagerly handled. Because of this, `LogicalRegionField` does bookkeeping for
 * potentially deferred materialization in its `mapped_` field, instead of directly examining the
 * outcome of materialization (i.e., the `Legion::PhysicalRegion` object). Doing so allows the
 * control code to check if a given `LogicalRegionField` has been materialized (which then requires
 * the runtime to run any consumer tasks in a blocking fashion) without flushing the scheduling
 * window.
 *
 * As of GH 2901, a read-only attachment is not considered as a materialization of the region field.
 * This means that the `mapped_` field won't be set to true upon read-only attachment. This is in
 * line with the contract of read-only attachment that any updates from downstream consumer tasks to
 * the region field don't need to be propagated back to the attachments.
 */
class LogicalRegionField : public legate::EnableSharedFromThis<LogicalRegionField> {
 public:
  class PhysicalState {
   public:
    [[nodiscard]] const Legion::PhysicalRegion& ensure_mapping(const Legion::LogicalRegion& region,
                                                               Legion::FieldID field_id,
                                                               legate::mapping::StoreTarget target);
    void set_physical_region(Legion::PhysicalRegion physical_region);
    /**
     * @brief Set an attachment to the physical state
     */
    void set_attachment(Attachment attachment);
    /**
     * @brief Mark in `has_pending_detach_` if the physical state has a pending detach operation.
     * The value is set to `true` if the attachment exists and is not read-only.
     */
    void set_has_pending_detach();
    /**
     * @brief Add a callback to run when the physical state is recycled.
     */
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
     * @brief Invoke all callbacks in this physical state. This function is idempotent (i.e.,
     * callbacks will be invoked only once no matter how many times this function is called.)
     */
    void invoke_callbacks();
    /**
     * @brief Deallocate the attachment of this physical state.
     *
     * @param wait_on_detach If `true`, the deallocation waits for the pending detach operation to
     * finish.
     */
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
  /**
   * @brief Update the value of `mapped_` of this region field.
   *
   * @param mapped A new value for `mapped_`
   */
  void set_mapped(bool mapped);
  void unmap();
  void attach(Legion::PhysicalRegion physical_region,
              InternalSharedPtr<ExternalAllocation> allocation);
  void attach(Legion::ExternalResources external_resources,
              std::vector<InternalSharedPtr<ExternalAllocation>> allocations);
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
  bool destroyed_out_of_order_{};

  // This object is updated in a deferred manner by a ReleaseRegionField operation
  InternalSharedPtr<PhysicalState> physical_state_{};
};

}  // namespace legate::detail

#include <legate/data/detail/logical_region_field.inl>
