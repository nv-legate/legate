/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/buffer.h>

#include <legate/data/detail/inline_storage.h>
#include <legate/data/inline_allocation.h>

#include <cstddef>
#include <utility>
#include <vector>

namespace legate::detail {

TaskLocalBuffer::TaskLocalBuffer(InternalSharedPtr<InlineStorage> storage,
                                 InternalSharedPtr<Type> type,
                                 // Domain is a POD type, no point in moving it
                                 const Domain& bounds)  // NOLINT(modernize-pass-by-value)
  : storage_{std::move(storage)}, type_{std::move(type)}, domain_{bounds}
{
}

mapping::StoreTarget TaskLocalBuffer::memory_kind() const { return storage_->target(); }

InlineAllocation TaskLocalBuffer::get_inline_allocation() const
{
  auto&& [base, byte_strides] = storage_->base_and_byte_strides();

  return {base, std::vector<std::size_t>(byte_strides.begin(), byte_strides.end()), memory_kind()};
}

}  // namespace legate::detail
