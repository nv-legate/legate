/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "legate/cuda/detail/nvtx.h"
#include "legate/task/detail/returned_exception.h"
#include "legate/task/task_context.h"
#include "legate/utilities/typedefs.h"

#include <optional>
#include <string_view>

namespace legate::detail {

void show_progress(const Legion::Task* task, Legion::Context ctx, Legion::Runtime* runtime);

void show_progress(const DomainPoint& index_point,
                   std::string_view task_name,
                   std::string_view provenance,
                   Legion::Context ctx      = Legion::Runtime::get_context(),
                   Legion::Runtime* runtime = Legion::Runtime::get_runtime());

namespace task_detail {

template <typename F>
[[nodiscard]] std::optional<ReturnedException> task_body(legate::TaskContext ctx,
                                                         VariantImpl variant_impl,
                                                         F&& get_task_name);

template <typename T, typename U>
[[nodiscard]] nvtx3::scoped_range make_nvtx_range(T&& get_task_name, U&& get_provenance);

}  // namespace task_detail

}  // namespace legate::detail

#include "legate/task/detail/task.inl"
