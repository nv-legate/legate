/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/logical_store.h>
#include <legate/utilities/detail/doxygen.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace legate {

class Shape;
class Type;

}  // namespace legate

namespace legate::experimental::io::kvikio {

// NOLINTNEXTLINE(readability-identifier-naming)
[[nodiscard]] LEGATE_EXPORT LogicalStore from_file_(const std::filesystem::path& file_path,
                                                    const Type& type);

[[nodiscard]] LEGATE_EXPORT LogicalStore
// NOLINTNEXTLINE(readability-identifier-naming)
from_file_(const std::filesystem::path& file_path,
           const Shape& shape,
           const Type& type,
           const std::vector<std::uint64_t>& tile_shape,
           std::optional<std::vector<std::uint64_t>> tile_start = {});

[[nodiscard]] LEGATE_EXPORT LogicalStore
// NOLINTNEXTLINE(readability-identifier-naming)
from_file_by_offsets_(const std::filesystem::path& file_path,
                      const Shape& shape,
                      const Type& type,
                      const std::vector<std::uint64_t>& offsets,
                      const std::vector<std::uint64_t>& tile_shape);

}  // namespace legate::experimental::io::kvikio
