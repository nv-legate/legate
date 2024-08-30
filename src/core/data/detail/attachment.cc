/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/data/detail/attachment.h"

#include "legate_defines.h"

#include "core/runtime/detail/runtime.h"

#include <exception>
#include <memory>
#include <type_traits>

namespace legate::detail {

Attachment::~Attachment() noexcept
{
  if (has_started()) {
    maybe_deallocate();
    return;
  }

  try {
    if (can_dealloc_.has_value() && can_dealloc_->exists()) {
      // FIXME: Leak the Future handle if the runtime has already shut down, as there's no hope that
      // this would be collected by the Legion runtime
      // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
      static_cast<void>(std::make_unique<Legion::Future>(*std::move(can_dealloc_)).release());
    }

    std::visit(  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
      [&](auto&& handle) {
        if (!handle.exists()) {
          return;
        }
        // FIXME: Leak the Legion handle if the runtime has already shut down, as there's no hope
        // that this would be collected by the Legion runtime
        static_cast<void>(
          std::make_unique<std::decay_t<decltype(handle)>>(std::forward<decltype(handle)>(handle))
            .release());
      },  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
      handle_);
  } catch (const std::exception& exn) {
    LEGATE_ABORT(exn.what());
  }
}

void Attachment::detach(bool unordered)
{
  if (!exists()) {
    return;
  }

  can_dealloc_.emplace(std::visit(
    [&](auto&& handle) {
      using T = std::decay_t<decltype(handle)>;
      if constexpr (std::is_same_v<T, Legion::PhysicalRegion>) {
        LEGATE_ASSERT(allocations_.size() == 1);
        return Runtime::get_runtime()->detach(
          handle, !allocations_.front()->read_only(), unordered);
      }
      if constexpr (std::is_same_v<T, Legion::ExternalResources>) {
        return Runtime::get_runtime()->detach(handle, false /*flush*/, unordered);
      }
    },
    handle_));
}

void Attachment::maybe_deallocate() noexcept
{
  if (!exists()) {
    return;
  }

  if (can_dealloc_.has_value() && can_dealloc_->exists()) {
    can_dealloc_->wait();
    can_dealloc_.reset();
  }

  for (auto&& allocation : allocations_) {
    allocation->maybe_deallocate();
  }
  allocations_.clear();
}

}  // namespace legate::detail
