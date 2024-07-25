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

#include "core/task/task_info.h"

namespace legate {

// NOLINTBEGIN(readability-identifier-naming)
template <typename T>
void TaskInfo::add_variant_(AddVariantKey,
                            Library library,
                            LegateVariantCode vid,
                            LegionVariantImpl<T> /*body*/,
                            Processor::TaskFuncPtr entry,
                            const VariantOptions* decl_options,
                            const std::map<LegateVariantCode, VariantOptions>& registration_options)
// NOLINTEND(readability-identifier-naming)
{
  // TODO(wonchanl): pass a null pointer as the body here as the function does not have the type
  // signature for Legate task variants. In the future we should extend VariantInfo so we can
  // distinguish Legate tasks from Legion tasks.
  add_variant_(AddVariantKey{},
               std::move(library),
               vid,
               VariantImpl{},
               entry,
               decl_options,
               registration_options);
}

}  // namespace legate
