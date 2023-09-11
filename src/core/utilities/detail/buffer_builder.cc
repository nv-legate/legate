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

#include "core/utilities/detail/buffer_builder.h"

namespace legate::detail {

BufferBuilder::BufferBuilder()
{
  // Reserve 4KB to minimize resizing while packing the arguments.
  buffer_.reserve(4096);
}

void BufferBuilder::pack_buffer(const void* src, size_t size)
{
  if (0 == size) return;
  auto off = buffer_.size();
  buffer_.resize(buffer_.size() + size);
  auto tgt = buffer_.data() + off;
  memcpy(tgt, src, size);
}

Legion::UntypedBuffer BufferBuilder::to_legion_buffer() const
{
  return Legion::UntypedBuffer(buffer_.data(), buffer_.size());
}

}  // namespace legate::detail
