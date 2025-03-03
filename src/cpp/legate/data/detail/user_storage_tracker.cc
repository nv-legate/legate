/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/user_storage_tracker.h>

namespace legate::detail {

UserStorageTracker::UserStorageTracker(const InternalSharedPtr<LogicalStore>& store)
  : storage_{[&] {
      auto&& storage = store->get_storage();
      return storage->get_root(storage).as_user_ptr();
    }()}
{
}

UserStorageTracker::~UserStorageTracker() noexcept
{
  if (storage_.user_ref_count() == 1) {
    storage_->free_early();
  }
}

}  // namespace legate::detail
