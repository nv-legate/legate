/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/allocator.h>

namespace legate {

template <typename T>
T* ScopedAllocator::allocate_type(std::size_t num_items)
{
  return static_cast<T*>(allocate_aligned(sizeof(T) * num_items, alignof(T)));
}

}  // namespace legate
