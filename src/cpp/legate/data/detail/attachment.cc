/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/attachment.h>

#include <legate_defines.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/type_traits.h>

#include <exception>
#include <memory>
#include <type_traits>

namespace legate::detail {

namespace {

// NOLINTBEGIN(clang-analyzer-cplusplus.NewDeleteLeaks)
template <typename T>
void intentionally_leak_handle(T&& handle)
{
  static_cast<void>(std::make_unique<T>(std::forward<T>(handle)).release());
}

// NOLINTEND(clang-analyzer-cplusplus.NewDeleteLeaks)

}  // namespace

Attachment::~Attachment() noexcept
{
  if (has_started()) {
    maybe_deallocate(true);
    return;
  }

  try {
    if (can_dealloc_.has_value() && can_dealloc_->exists()) {
      // FIXME: Leak the Future handle if the runtime has already shut down, as there's no hope that
      // this would be collected by the Legion runtime
      intentionally_leak_handle(*std::move(can_dealloc_));
    }

    std::visit(
      [&](auto&& handle) {
        if (!handle.exists()) {
          return;
        }
        // FIXME: Leak the Legion handle if the runtime has already shut down, as there's no hope
        // that this would be collected by the Legion runtime
        //
        // We explicitly want to move out of the variant here, so moving an lvalue is desired.
        // NOLINTNEXTLINE(bugprone-move-forwarding-reference)
        intentionally_leak_handle(std::move(handle));
      },
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
    Overload{[&](const Legion::PhysicalRegion& region) {
               LEGATE_ASSERT(allocations_.size() == 1);
               return Runtime::get_runtime().detach(
                 region, !allocations_.front()->read_only(), unordered);
             },
             [&](const Legion::ExternalResources& resources) {
               return Runtime::get_runtime().detach(resources, /*flush=*/false, unordered);
             }},
    handle_));
}

void Attachment::maybe_deallocate(bool wait_on_detach) noexcept
{
  if (!exists()) {
    return;
  }

  if (can_dealloc_.has_value() && can_dealloc_->exists() && wait_on_detach) {
    can_dealloc_->wait();
  }
  can_dealloc_.reset();

  for (auto&& allocation : allocations_) {
    allocation->maybe_deallocate();
  }
  allocations_.clear();
}

}  // namespace legate::detail
