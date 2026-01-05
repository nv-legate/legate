/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/cuda/detail/nvtx.h>
#include <legate/task/detail/returned_exception.h>
#include <legate/task/task_context.h>
#include <legate/utilities/typedefs.h>

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

#include <legate/task/detail/task.inl>
