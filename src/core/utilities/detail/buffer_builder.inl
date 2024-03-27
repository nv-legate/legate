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

// Useful for IDEs
#include "core/utilities/detail/buffer_builder.h"

namespace legate::detail {

template <typename T>
void BufferBuilder::pack(const T& value)
{
  pack_buffer(reinterpret_cast<const int8_t*>(std::addressof(value)), sizeof(T), alignof(T));
}

template <typename T>
void BufferBuilder::pack(const std::vector<T>& values)
{
  const std::uint32_t size = values.size();
  pack(size);
  pack_buffer(values.data(), size * sizeof(T), alignof(T));
}

template <typename T>
void BufferBuilder::pack(const tuple<T>& values)
{
  pack(values.data());
}

}  // namespace legate::detail
