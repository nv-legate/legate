/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/task_info.h>

namespace legate::detail {

inline detail::ZStringView TaskInfo::name() const { return task_name_; }

inline const InternalSharedPtr<TaskConfig>& TaskInfo::task_config() const { return task_config_; }

inline const std::map<VariantCode, VariantInfo>& TaskInfo::variants_() const
{
  return task_variants_;
}

inline std::map<VariantCode, VariantInfo>& TaskInfo::variants_() { return task_variants_; }

}  // namespace legate::detail
