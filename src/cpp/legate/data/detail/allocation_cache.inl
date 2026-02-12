/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/allocation_cache.h>

#include <cstdint>

namespace legate::detail {

inline AllocationCache::AllocationCache(Allocator allocator,
                                        Deleter deleter,
                                        std::uint64_t max_size)
  : allocator_{std::move(allocator)}, deleter_{std::move(deleter)}, max_total_size_{max_size}
{
}

}  // namespace legate::detail
