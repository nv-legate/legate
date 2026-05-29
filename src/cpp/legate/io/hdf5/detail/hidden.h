/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/logical_store.h>

#include <filesystem>
#include <string_view>

namespace legate::io::hdf5 {

// Hidden methods to drop legate arrays from the Python API

// NOLINTNEXTLINE(readability-identifier-naming)
[[nodiscard]] LEGATE_EXPORT LogicalStore from_file_(const std::filesystem::path& file_path,
                                                    std::string_view dataset_name);

// NOLINTNEXTLINE(readability-identifier-naming)
LEGATE_EXPORT void to_file_(const LogicalStore& store,
                            std::filesystem::path file_path,
                            std::string_view dataset_name);

}  // namespace legate::io::hdf5
