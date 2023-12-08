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

#include "core/data/logical_array.h"

namespace legate {

inline LogicalArray::LogicalArray(InternalSharedPtr<detail::LogicalArray> impl)
  : impl_(std::move(impl))
{
}

inline const SharedPtr<detail::LogicalArray>& LogicalArray::impl() const { return impl_; }

// ==========================================================================================

inline ListLogicalArray::ListLogicalArray(InternalSharedPtr<detail::LogicalArray> impl)
  : LogicalArray{std::move(impl)}
{
}

// ==========================================================================================

inline StringLogicalArray::StringLogicalArray(InternalSharedPtr<detail::LogicalArray> impl)
  : LogicalArray{std::move(impl)}
{
}

}  // namespace legate
