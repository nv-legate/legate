/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/streaming/generation.h>

#include <legate/utilities/detail/buffer_builder.h>

namespace legate::detail {

void StreamingGeneration::pack(BufferBuilder& buffer) const
{
  buffer.pack(generation);
  buffer.pack(size);
}

}  // namespace legate::detail
