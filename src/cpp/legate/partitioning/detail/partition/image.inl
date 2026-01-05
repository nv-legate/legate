/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/partition/image.h>

namespace legate::detail {

inline bool Image::is_complete_for(const detail::Storage& /*storage*/) const
{
  // Completeness check for image partitions is expensive, so we give a sound answer
  return false;
}

inline bool Image::is_convertible() const { return false; }

inline bool Image::is_invertible() const { return false; }

inline const InternalSharedPtr<detail::LogicalStore>& Image::func() const { return func_; }

}  // namespace legate::detail
