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

#include "core/task/detail/return.h"

#include <utility>

namespace legate::detail {

inline size_t ReturnValue::size() const { return size_; }

inline bool ReturnValue::is_device_value() const { return is_device_value_; }

// ==========================================================================================

inline bool ReturnedException::raised() const { return raised_; }

// ==========================================================================================

inline size_t ReturnValues::legion_buffer_size() const { return buffer_size_; }

inline ReturnValue ReturnValues::operator[](int32_t idx) const { return return_values_[idx]; }

}  // namespace legate::detail
