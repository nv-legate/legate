/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/remap_request.h>

#include <utility>

namespace legate::detail {

inline RemapRequest::RemapRequest(InternalSharedPtr<InlineStorage> source,
                                  InternalSharedPtr<InlineStorage> target,
                                  std::uint32_t field_size)
  : source_{std::move(source)}, target_{std::move(target)}, field_size_{field_size}
{
}

}  // namespace legate::detail
