/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/task_context.h>

namespace legate {

inline TaskContext::TaskContext(detail::TaskContext* impl) : impl_{impl} {}

inline detail::TaskContext* TaskContext::impl() const { return impl_; }

}  // namespace legate
