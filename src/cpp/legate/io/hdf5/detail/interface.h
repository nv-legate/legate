/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/logical_array.h>

#include <filesystem>
#include <string_view>

namespace legate::io::hdf5::detail {

[[nodiscard]] LogicalArray from_file(const std::filesystem::path& file_path,
                                     std::string_view dataset_name);

}  // namespace legate::io::hdf5::detail
