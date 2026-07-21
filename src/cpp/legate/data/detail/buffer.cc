/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/buffer.h>

#include <legate/data/detail/inline_storage.h>
#include <legate/data/inline_allocation.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace legate::detail {

TaskLocalBuffer::TaskLocalBuffer(InternalSharedPtr<InlineStorage> storage,
                                 InternalSharedPtr<Type> type,
                                 // Rect is a POD type, no point in moving it
                                 const Rect& bounds)  // NOLINT(modernize-pass-by-value)
  : storage_{std::move(storage)}, type_{std::move(type)}, domain_{bounds}
{
}

mapping::StoreTarget TaskLocalBuffer::memory_kind() const { return storage_->target(); }

InlineAllocation TaskLocalBuffer::get_inline_allocation() const
{
  // `InlineAllocation` reports byte strides, so scale the storage's element strides by the field
  // size here (the storage keeps strides in element units).
  auto&& [base, elem_strides] = storage_->base_and_element_strides();
  const auto field_size       = type_->size();

  std::vector<std::size_t> byte_strides;

  byte_strides.reserve(elem_strides.dim());
  for (std::uint32_t i = 0; i < elem_strides.dim(); ++i) {
    byte_strides.push_back(static_cast<std::size_t>(elem_strides[i]) * field_size);
  }
  return {base, std::move(byte_strides), memory_kind()};
}

}  // namespace legate::detail
