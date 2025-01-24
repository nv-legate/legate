/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/task/detail/task_return_layout.h>

namespace legate::detail {

inline TaskReturnLayoutForUnpack::TaskReturnLayoutForUnpack(std::size_t starting_offset)
  : current_offset_{starting_offset}
{
}

inline std::size_t TaskReturnLayoutForUnpack::total_size() const { return current_offset_; }

inline TaskReturnLayoutForPack::const_iterator TaskReturnLayoutForPack::begin() const
{
  return offsets_.begin();
}

inline TaskReturnLayoutForPack::const_iterator TaskReturnLayoutForPack::end() const
{
  return offsets_.end();
}

}  // namespace legate::detail
