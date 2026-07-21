/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/task_info.h>

namespace legate {

inline const SharedPtr<detail::TaskInfo>& TaskInfo::impl() const { return impl_; }

inline TaskInfo::TaskInfo(InternalSharedPtr<detail::TaskInfo> impl) : impl_{std::move(impl)} {}

}  // namespace legate
