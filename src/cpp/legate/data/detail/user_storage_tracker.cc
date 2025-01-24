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
