/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/task_info.h>

namespace legate::detail {

inline TaskInfo::TaskInfo(std::string task_name) : task_name_{std::move(task_name)} {}

inline detail::ZStringView TaskInfo::name() const { return task_name_; }

inline const std::map<VariantCode, VariantInfo>& TaskInfo::variants_() const
{
  return task_variants_;
}

inline std::map<VariantCode, VariantInfo>& TaskInfo::variants_() { return task_variants_; }

}  // namespace legate::detail
