/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/return_value.h>

namespace legate::detail {

inline std::size_t ReturnValue::size() const { return size_; }

inline std::size_t ReturnValue::alignment() const { return alignment_; }

inline bool ReturnValue::is_device_value() const { return is_device_value_; }

}  // namespace legate::detail
