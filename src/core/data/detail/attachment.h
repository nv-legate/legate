/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/data/detail/external_allocation.h"
#include "core/utilities/internal_shared_ptr.h"

#include "legion.h"

#include <memory>
#include <vector>

namespace legate::detail {

class Attachment {
 public:
  Attachment()                                                      = default;
  virtual ~Attachment()                                             = default;
  Attachment(const Attachment&)                                     = delete;
  Attachment& operator=(const Attachment&)                          = delete;
  [[nodiscard]] virtual Legion::Future detach(bool unordered) const = 0;
  virtual void maybe_deallocate() noexcept                          = 0;
};

class SingleAttachment final : public Attachment {
 public:
  SingleAttachment(Legion::PhysicalRegion* physical_region,
                   InternalSharedPtr<ExternalAllocation> allocation);
  ~SingleAttachment() final;
  [[nodiscard]] Legion::Future detach(bool unordered) const override;
  void maybe_deallocate() noexcept override;

 private:
  // This physical region is owned by the logical region field embedding this attachment
  Legion::PhysicalRegion* physical_region_{};
  InternalSharedPtr<ExternalAllocation> allocation_{};
};

class IndexAttachment final : public Attachment {
 public:
  IndexAttachment(const Legion::ExternalResources& external_resources,
                  std::vector<InternalSharedPtr<ExternalAllocation>> allocations);
  ~IndexAttachment() final;
  [[nodiscard]] Legion::Future detach(bool unordered) const override;
  void maybe_deallocate() noexcept override;

 private:
  // Unlike the single attachment, the index attachment owns this ExternalResources object
  std::unique_ptr<Legion::ExternalResources> external_resources_{};
  std::vector<InternalSharedPtr<ExternalAllocation>> allocations_{};
};

}  // namespace legate::detail

#include "core/data/detail/attachment.inl"
