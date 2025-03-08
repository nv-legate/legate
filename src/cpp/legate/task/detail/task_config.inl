/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/task_config.h>

namespace legate::detail {

inline LocalTaskID TaskConfig::task_id() const { return task_id_; }

inline const std::optional<InternalSharedPtr<TaskSignature>>& TaskConfig::signature() const
{
  return signature_;
}

inline const std::optional<VariantOptions>& TaskConfig::variant_options() const
{
  return variant_options_;
}

}  // namespace legate::detail
