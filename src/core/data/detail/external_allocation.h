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

#include "core/mapping/mapping.h"

#include "legion.h"

#include "realm/instance.h"

#include <memory>
#include <optional>

namespace legate::detail {

class ExternalAllocation {
 public:
  using Deleter = void (*)(void*) noexcept;

  ExternalAllocation(bool read_only,
                     mapping::StoreTarget target,
                     void* ptr,
                     size_t size,
                     std::unique_ptr<Realm::ExternalInstanceResource> resource,
                     std::optional<Deleter> deleter = std::nullopt);

  [[nodiscard]] bool read_only() const;
  [[nodiscard]] mapping::StoreTarget target() const;
  [[nodiscard]] void* ptr() const;
  [[nodiscard]] size_t size() const;
  [[nodiscard]] const Realm::ExternalInstanceResource* resource() const;
  void maybe_deallocate();

  // detail::ExternalAllocation should never be copied
  ExternalAllocation(const ExternalAllocation& other)            = delete;
  ExternalAllocation& operator=(const ExternalAllocation& other) = delete;

 private:
  bool read_only_{};
  mapping::StoreTarget target_{mapping::StoreTarget::SYSMEM};
  void* ptr_{};
  size_t size_{};
  std::unique_ptr<Realm::ExternalInstanceResource> resource_{};
  std::optional<Deleter> deleter_{};
};

}  // namespace legate::detail

#include "core/data/detail/external_allocation.inl"
