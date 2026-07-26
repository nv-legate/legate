/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/remap_request.h>

#include <utility>

namespace legate::detail {

inline RemapRequest::RemapRequest(InternalSharedPtr<Buffer> source,
                                  InternalSharedPtr<Buffer> target)
  : source_{std::move(source)}, target_{std::move(target)}
{
}

}  // namespace legate::detail
