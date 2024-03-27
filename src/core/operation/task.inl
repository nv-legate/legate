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

#include "core/operation/task.h"

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
