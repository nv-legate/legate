/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/io/hdf5/interface.h>

#include <legate_defines.h>

#include <legate/io/hdf5/detail/interface.h>
#include <legate/utilities/detail/traced_exception.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

namespace legate::io::hdf5 {

InvalidDataSetError::InvalidDataSetError(const std::string& what,
                                         std::filesystem::path path,
                                         std::string dataset_name)
  : invalid_argument{what}, path_{std::move(path)}, dataset_name_{std::move(dataset_name)}
{
}

const std::filesystem::path& InvalidDataSetError::path() const noexcept { return path_; }

std::string_view InvalidDataSetError::dataset_name() const noexcept { return dataset_name_; }

// ==========================================================================================

LogicalArray from_file(const std::filesystem::path& file_path, std::string_view dataset_name)
{
  if (!LEGATE_DEFINED(LEGATE_USE_HDF5)) {
    throw legate::detail::TracedException<std::runtime_error>{
      "Legate was not configured with HDF5 support. Please reconfigure Legate with HDF5 support to "
      "use this API."};
  }
  return detail::from_file(file_path, dataset_name);
}

}  // namespace legate::io::hdf5
