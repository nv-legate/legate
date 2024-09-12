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

#include "legate/task/task.h"

#include "legate/task/detail/legion_task_body.h"
#include "legate/utilities/typedefs.h"

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
