/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
