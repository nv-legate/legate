/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

/**
 * @brief Write a LogicalArray to disk as a HDF5 dataset.
 *
 * See `legate::io::hdf5::to_file()` for further discussion on the semantics of this
 * routine. This 2-step exists purely to isolate the HDF5-specific symbols from the interface
 * (which must compile regardless of whether HDF5 is available).
 *
 * @param array The array to store.
 * @param file_path The resulting HDF5 file.
 * @param dataset_name The HDF5 dataset name to store the array under. See
 * https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html for further
 * discussion on datasets.
 */
void to_file(const LogicalArray& array,
             std::filesystem::path file_path,
             std::string_view dataset_name);

}  // namespace legate::io::hdf5::detail
