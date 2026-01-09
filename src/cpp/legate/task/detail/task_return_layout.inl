/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
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
