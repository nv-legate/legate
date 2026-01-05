/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_store.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/shared_ptr.h>

namespace legate::detail {

class UserStorageTracker {
 public:
  explicit UserStorageTracker(const InternalSharedPtr<LogicalStore>& store);

  ~UserStorageTracker() noexcept;

  UserStorageTracker(const UserStorageTracker&)                = default;
  UserStorageTracker& operator=(const UserStorageTracker&)     = default;
  UserStorageTracker(UserStorageTracker&&) noexcept            = default;
  UserStorageTracker& operator=(UserStorageTracker&&) noexcept = default;

 private:
  legate::SharedPtr<Storage> storage_{};
};

}  // namespace legate::detail
