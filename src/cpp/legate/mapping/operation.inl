/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/operation.h>

namespace legate::mapping {

inline Task::Task(const detail::Task* impl) : pimpl_{impl} {}

inline const detail::Task* Task::impl() const noexcept { return pimpl_; }

}  // namespace legate::mapping
