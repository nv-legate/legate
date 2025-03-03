/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/task.h>
#include <legate/utilities/typedefs.h>

namespace legate::detail {

void inline_task_body(const Task& task, VariantCode variant_code, VariantImpl variant_impl);

}  // namespace legate::detail
