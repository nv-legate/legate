/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/task.h>
#include <legate/utilities/typedefs.h>

namespace legate::detail {

class Strategy;

void inline_task_body(const TaskBase& task,
                      VariantCode variant_code,
                      VariantImpl variant_impl,
                      const Strategy& strategy);

}  // namespace legate::detail
