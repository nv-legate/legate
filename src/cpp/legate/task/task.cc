/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/task.h>

#include <legate/task/detail/legion_task_body.h>
#include <legate/utilities/typedefs.h>

#include <optional>

namespace legate::detail {

void task_wrapper(VariantImpl variant_impl,
                  VariantCode variant_kind,
                  std::optional<std::string_view> task_name,
                  const void* args,
                  std::size_t arglen,
                  const void* /*userdata*/,
                  std::size_t /*userlen*/,
                  Processor p)
{
  legion_task_body(variant_impl, variant_kind, std::move(task_name), args, arglen, std::move(p));
}

}  // namespace legate::detail
