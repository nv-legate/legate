/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/external_allocation.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <legion.h>

#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace legate::detail {

class Attachment {
 public:
  Attachment() = default;
  // Single attach case
  Attachment(Legion::PhysicalRegion physical_region,
             InternalSharedPtr<ExternalAllocation> allocation);
  // Index attach case
  Attachment(Legion::ExternalResources external_resources,
             std::vector<InternalSharedPtr<ExternalAllocation>> allocations);

  ~Attachment() noexcept;
  Attachment(const Attachment&)            = delete;
  Attachment& operator=(const Attachment&) = delete;
  Attachment(Attachment&&)                 = default;
  Attachment& operator=(Attachment&&)      = default;
  void detach(bool unordered);
  // This function block-waits until the outstanding detach operation is finished. This function
  // however does not flush the scheduling window and it's the caller's responsibility to make sure
  // the deferred detach operation for this attachment is scheduled by calling
  // `Runtime::flush_scheduling_window`.
  void maybe_deallocate(bool wait_on_detach) noexcept;
  [[nodiscard]] bool exists() const noexcept;

 protected:
  std::optional<Legion::Future> can_dealloc_{};
  std::variant<Legion::PhysicalRegion, Legion::ExternalResources> handle_{};
  std::vector<InternalSharedPtr<ExternalAllocation>> allocations_{};
};

}  // namespace legate::detail

#include <legate/data/detail/attachment.inl>
