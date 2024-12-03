/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/data/logical_array.h>
#include <legate/utilities/detail/doxygen.h>

#include <filesystem>
#include <string_view>

/**
 * @file
 * @brief Interface for HDF5 I/O
 */

namespace legate::experimental::io::hdf5 {

/**
 * @addtogroup io-hdf5
 * @{
 */

/**
 * @brief An exception thrown when a HDF5 datatype could not be converted to a Type.
 */
class UnsupportedHDF5DataTypeError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

/**
 * @brief An exception thrown when an invalid dataset is encountered in an HDF5 file.
 */
class InvalidDataSetError : public std::invalid_argument {
 public:
  /**
   * @brief Construct an InvalidDataSetError
   *
   * @param what The exception string to forward to the constructor of std::invalid_argument.
   * @param path The path to the HDF5 file containing the dataset.
   * @param dataset_name The name of the offending dataset.
   */
  InvalidDataSetError(const std::string& what,
                      std::filesystem::path path,
                      std::string dataset_name);

  /**
   * @brief Get the path to the file containing the dataset.
   *
   * @return The path to the file containing the dataset.
   */
  [[nodiscard]] const std::filesystem::path& path() const noexcept;

  /**
   * @brief Get the name of the dataset.
   *
   * @return The name of the dataset.
   */
  [[nodiscard]] std::string_view dataset_name() const noexcept;

 private:
  std::filesystem::path path_{};
  std::string dataset_name_{};
};

/**
 * @brief Load a HDF5 dataset into a LogicalArray.
 *
 * @param file_path The path to the file to load.
 * @param dataset_name The name of the HDF5 dataset to load from the file.
 *
 * @return LogicalArray The loaded array.
 *
 * @throws std::system_error If file_path does not exist.
 * @throws UnusupportedHDF5DataType If the data type cannot be converted to a Type.
 * @throws InvalidDataSetError If the dataset is invalid, or is not found.
 *
 * @warning This API is experimental. A future release may change or remove this API without
 * warning, deprecation period, or notice. The user is nevertheless encouraged to use this API,
 * and submit any feedback to legate@nvidia.com.
 */
[[nodiscard]] LogicalArray from_file(const std::filesystem::path& file_path,
                                     std::string_view dataset_name);

/** @} */

}  // namespace legate::experimental::io::hdf5
