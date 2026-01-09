/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/external_allocation.h>
#include <legate/mapping/mapping.h>

#include <legion.h>
#include <realm/instance.h>

#include <memory>
#include <optional>

namespace legate::detail {

class ExternalAllocation {
 public:
  using Deleter = legate::ExternalAllocation::Deleter;

  ExternalAllocation(bool read_only,
                     mapping::StoreTarget target,
                     void* ptr,
                     std::size_t size,
                     std::unique_ptr<Realm::ExternalInstanceResource> resource,
                     std::optional<Deleter> deleter = std::nullopt);

  [[nodiscard]] bool read_only() const;
  [[nodiscard]] mapping::StoreTarget target() const;
  [[nodiscard]] void* ptr() const;
  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] const Realm::ExternalInstanceResource* resource() const;
  void maybe_deallocate();

  // detail::ExternalAllocation should never be copied
  ExternalAllocation(const ExternalAllocation& other)            = delete;
  ExternalAllocation& operator=(const ExternalAllocation& other) = delete;

 private:
  bool read_only_{};
  mapping::StoreTarget target_{mapping::StoreTarget::SYSMEM};
  void* ptr_{};
  std::size_t size_{};
  std::unique_ptr<Realm::ExternalInstanceResource> resource_{};
  std::optional<Deleter> deleter_{};
};

}  // namespace legate::detail

#include <legate/data/detail/external_allocation.inl>
