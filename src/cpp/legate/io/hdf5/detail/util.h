/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/io/hdf5/detail/hdf5_wrapper.h>

#include <highfive/H5File.hpp>

#include <cstddef>
#include <string>

namespace legate::io::hdf5::detail {

[[nodiscard]] HighFive::File open_hdf5_file(
  const wrapper::HDF5MaybeLockGuard&,
  const std::string& filepath,
  bool gds_on,
  HighFive::File::AccessMode open_mode = HighFive::File::ReadOnly);

}  // namespace legate::io::hdf5::detail
