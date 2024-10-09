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

#pragma once

#include "legate/data/detail/logical_store.h"
#include "legate/utilities/internal_shared_ptr.h"
#include "legate/utilities/shared_ptr.h"

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
