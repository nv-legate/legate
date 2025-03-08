/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/task_info.h>

namespace legate {

inline const SharedPtr<detail::TaskInfo>& TaskInfo::impl() const { return impl_; }

inline TaskInfo::TaskInfo(InternalSharedPtr<detail::TaskInfo> impl) : impl_{std::move(impl)} {}

template <typename T>
void TaskInfo::add_variant_(AddVariantKey,  // NOLINT(readability-identifier-naming)
                            const Library& library,
                            VariantCode vid,
                            LegionVariantImpl<T> /*body*/,
                            Processor::TaskFuncPtr entry,
                            const TaskConfig& task_config,
                            const VariantOptions* decl_options,
                            const std::map<VariantCode, VariantOptions>& registration_options)
{
  // TODO(wonchanl): pass a null pointer as the body here as the function does not have the type
  // signature for Legate task variants. In the future we should extend VariantInfo so we can
  // distinguish Legate tasks from Legion tasks.
  add_variant_(AddVariantKey{},
               library,
               vid,
               VariantImpl{},
               entry,
               task_config,
               decl_options,
               registration_options);
}

}  // namespace legate
