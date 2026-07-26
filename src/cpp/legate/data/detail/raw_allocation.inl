/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/raw_allocation.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

namespace legate::detail {

inline RawAllocation::RawAllocation(bool read_only,
                                    mapping::StoreTarget target,
                                    void* ptr,
                                    std::size_t size,
                                    std::optional<Deleter> deleter)
  : read_only_{read_only}, target_{target}, ptr_{ptr}, size_{size}, deleter_{std::move(deleter)}
{
}

inline bool RawAllocation::read_only() const { return read_only_; }

inline mapping::StoreTarget RawAllocation::target() const { return target_; }

inline void* RawAllocation::ptr() const { return ptr_; }

inline std::size_t RawAllocation::size() const { return size_; }

}  // namespace legate::detail
