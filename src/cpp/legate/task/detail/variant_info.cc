/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/variant_info.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <cstdint>

namespace legate::detail {

std::string VariantInfo::to_string() const
{
  return fmt::format("{:x}, {}", reinterpret_cast<std::uintptr_t>(body), fmt::streamed(options));
}

}  // namespace legate::detail
