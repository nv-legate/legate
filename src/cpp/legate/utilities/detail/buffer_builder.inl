/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Useful for IDEs
#include <legate/utilities/detail/buffer_builder.h>

namespace legate::detail {

template <typename T>
void BufferBuilder::pack(const T& value)
{
  pack_buffer(reinterpret_cast<const std::int8_t*>(std::addressof(value)),
              sizeof(T),  // NOLINT(bugprone-sizeof-expression)
              alignof(T));
}

template <typename T>
void BufferBuilder::pack(Span<const T> values)
{
  const std::uint32_t size = values.size();
  pack(size);
  pack_buffer(values.data(), size * sizeof(T), alignof(T));
}

}  // namespace legate::detail
