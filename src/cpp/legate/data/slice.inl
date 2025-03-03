/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/slice.h>

namespace legate {

inline Slice::Slice(std::optional<std::int64_t> _start, std::optional<std::int64_t> _stop)
  : start{std::move(_start)}, stop{std::move(_stop)}
{
}

}  // namespace legate
