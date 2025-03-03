/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/task.h>

namespace legate {

template <typename T, typename Enable>
void AutoTask::add_scalar_arg(T&& value)
{
  add_scalar_arg(Scalar{std::forward<T>(value)});
}

template <typename T, typename Enable>
void ManualTask::add_scalar_arg(T&& value)
{
  add_scalar_arg(Scalar{std::forward<T>(value)});
}

}  // namespace legate
